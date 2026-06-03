import time
import json
import datetime
from pathlib import Path

import torch
import numpy as np

from src.misc import dist
from src.data import get_coco_api_from_dataset
from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt


class DetSolver(BaseSolver):

    def _ensure_output_dir(self):

        if self.output_dir is None or str(self.output_dir) == "":
            print("[DetSolver][WARN] output_dir 为空，将使用默认 ./output")
            self.output_dir = Path("./output")
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "eval").mkdir(parents=True, exist_ok=True)


    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg


        print(f"[DetSolver][CFG] epoches={getattr(args, 'epoches', None)}, "
              f"log_step={getattr(args, 'log_step', None)}, "
              f"checkpoint_step={getattr(args, 'checkpoint_step', None)}, "
              f"clip_max_norm={getattr(args, 'clip_max_norm', None)}")
        print(f"[DetSolver][CFG] use_amp={getattr(args, 'use_amp', None)}, "
              f"use_ema={getattr(args, 'use_ema', None)}")


        self._ensure_output_dir()
        print(f"[DetSolver] output_dir = {self.output_dir.resolve()}")


        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('[DetSolver] number of trainable params:', n_parameters)


        def _profile_flops_once(model, device, img_size=1024):

            m = model.module if hasattr(model, "module") else model
            x = torch.zeros(1, 3, img_size, img_size, device=device)

            was_training = m.training
            try:
                m.eval()
                with torch.no_grad():

                    try:
                        from thop import profile as thop_profile
                        flops, _params = thop_profile(m, inputs=(x,), verbose=False)
                        return float(flops) / 1e9, None
                    except Exception:
                        pass


                    try:
                        from fvcore.nn import FlopCountAnalysis
                        flops = FlopCountAnalysis(m, x).total()
                        return float(flops) / 1e9, None
                    except Exception as e:
                        return None, f"Profiling failed with thop and fvcore: {e}"
            finally:
                if was_training:
                    m.train()

        flops_g = None
        if dist.is_main_process():
            flops_g, flops_err = _profile_flops_once(self.model, self.device, img_size=1024)
            if flops_g is None:
                print(f"[DetSolver][WARN] FLOPs profiling unavailable, FLOPs_G will be None. Reason: {flops_err}")
            else:
                print(f"[DetSolver] Profiled FLOPs for 1024x1024: {flops_g:.4f} G")


        print("[DetSolver] building COCO API for validation dataset ...")
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        print("[DetSolver] COCO API ready.")

        best_stat = {'epoch': -1, }
        start_time = time.time()


        epoch_history = []
        train_loss_history = []
        val_acc_history = []
        val_recall_history = []

        try:
            train_len = len(self.train_dataloader)
        except Exception:
            train_len = -1
        try:
            val_len = len(self.val_dataloader)
        except Exception:
            val_len = -1
        print(f"[DetSolver] len(train_dataloader)={train_len}, len(val_dataloader)={val_len}")
        if val_len == 0:
            print("[DetSolver][WARN] 验证集 dataloader 没有任何批次（len=0），将导致评估阶段几乎“瞬间结束”。"
                  "请检查 val_dataloader 的 batch_size / drop_last / sampler 配置。")

        def _calc_mf1_from_coco(coco_evaluator):

            try:
                if coco_evaluator is None or "bbox" not in coco_evaluator.coco_eval:
                    return None
                bbox_eval = coco_evaluator.coco_eval["bbox"]
                if bbox_eval is None or bbox_eval.eval is None:
                    return None

                precision = bbox_eval.eval.get("precision", None)
                if precision is None:
                    return None


                T, R, K, A, M = precision.shape


                rec_thrs = getattr(getattr(bbox_eval, "params", None), "recThrs", None)
                if rec_thrs is None:
                    rec_thrs = np.linspace(0.0, 1.0, R, dtype=np.float32)
                else:
                    rec_thrs = np.asarray(rec_thrs, dtype=np.float32)


                p = precision[:, :, :, 0, -1]
                r = np.broadcast_to(rec_thrs.reshape(1, R, 1), p.shape)

                valid = p > -1
                denom = p + r
                denom = np.where(denom == 0, 1e-12, denom)

                f1 = np.where(valid, (2 * p * r) / denom, np.nan)


                f1_max_over_r = np.nanmax(f1, axis=1)


                f1_mean_over_t = np.nanmean(f1_max_over_r, axis=0)

                mf1 = np.nanmean(f1_mean_over_t)
                if np.isnan(mf1):
                    return None
                return float(mf1)
            except Exception:
                return None


        for epoch in range(self.last_epoch + 1, args.epoches):


            if dist.is_dist_available_and_initialized():
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch)

            print(f"\n[DetSolver] ======= EPOCH {epoch} / {args.epoches-1} =======")
            t0 = time.time()
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader,
                self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step,
                ema=self.ema, scaler=self.scaler,
                debug=getattr(args, 'train_debug', False),
                debug_steps=getattr(args, 'train_debug_steps', 3)
            )
            t1 = time.time()
            print(f"[DetSolver] epoch {epoch} train_one_epoch time: {t1 - t0:.2f}s")


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']

                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:

                    if dist.is_main_process():
                        print(f"[DetSolver] saving checkpoint to {checkpoint_path}")
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)


            module = self.ema.module if self.ema else self.model


            print(f"[DetSolver] evaluating on validation set ...")
            te0 = time.time()
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir
            )
            te1 = time.time()
            print(f"[DetSolver] evaluate time: {te1 - te0:.2f}s")


            epoch_history.append(epoch)


            if 'loss' in train_stats:
                train_loss_history.append(float(train_stats['loss']))
            else:

                train_loss_history.append(float(next(iter(train_stats.values()))))


            if 'coco_eval_bbox' in test_stats:
                coco_stats = test_stats['coco_eval_bbox']

                val_acc_history.append(float(coco_stats[0]))

                val_recall_history.append(float(coco_stats[8]))
            else:

                val_acc_history.append(0.0)
                val_recall_history.append(0.0)


            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('[DetSolver] best_stat: ', best_stat)


            coco_eval_bbox_raw = None

            AP_50_95 = None
            AP_50 = None
            AP_75 = None
            AP_small = None
            AP_medium = None
            AP_large = None

            AR_1 = None
            AR_10 = None
            AR_100 = None
            AR_small = None
            AR_medium = None
            AR_large = None

            if 'coco_eval_bbox' in test_stats:
                coco_stats = test_stats['coco_eval_bbox']

                try:
                    coco_eval_bbox_raw = [float(x) for x in coco_stats]
                except Exception:
                    coco_eval_bbox_raw = coco_stats


                try: AP_50_95 = float(coco_stats[0])
                except Exception: AP_50_95 = None
                try: AP_50 = float(coco_stats[1])
                except Exception: AP_50 = None
                try: AP_75 = float(coco_stats[2])
                except Exception: AP_75 = None
                try: AP_small = float(coco_stats[3])
                except Exception: AP_small = None
                try: AP_medium = float(coco_stats[4])
                except Exception: AP_medium = None
                try: AP_large = float(coco_stats[5])
                except Exception: AP_large = None

                try: AR_1 = float(coco_stats[6])
                except Exception: AR_1 = None
                try: AR_10 = float(coco_stats[7])
                except Exception: AR_10 = None
                try: AR_100 = float(coco_stats[8])
                except Exception: AR_100 = None
                try: AR_small = float(coco_stats[9])
                except Exception: AR_small = None
                try: AR_medium = float(coco_stats[10])
                except Exception: AR_medium = None
                try: AR_large = float(coco_stats[11])
                except Exception: AR_large = None


            mf1 = _calc_mf1_from_coco(coco_evaluator)
            params_m = float(n_parameters) / 1e6

            acc_pct = (AP_50 * 100.0) if (AP_50 is not None) else None
            map_50_95_pct = (AP_50_95 * 100.0) if (AP_50_95 is not None) else None
            map_50_pct = (AP_50 * 100.0) if (AP_50 is not None) else None

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,


                'Acc_pct': acc_pct,
                'mF1': mf1,
                'mAP_50_95_pct': map_50_95_pct,
                'mAP_50_pct': map_50_pct,
                'FLOPs_G': flops_g,
                'Params_M': params_m,


                'coco_eval_bbox_raw': coco_eval_bbox_raw,

                'AP_50_95': AP_50_95,
                'AP_50': AP_50,
                'AP_75': AP_75,
                'AP_small': AP_small,
                'AP_medium': AP_medium,
                'AP_large': AP_large,

                'AR_1': AR_1,
                'AR_10': AR_10,
                'AR_100': AR_100,
                'AR_small': AR_small,
                'AR_medium': AR_medium,
                'AR_large': AR_large,

                'AP_50_95_pct': (AP_50_95 * 100.0) if (AP_50_95 is not None) else None,
                'AP_50_pct': (AP_50 * 100.0) if (AP_50 is not None) else None,
                'AP_75_pct': (AP_75 * 100.0) if (AP_75 is not None) else None,
                'AP_small_pct': (AP_small * 100.0) if (AP_small is not None) else None,
                'AP_medium_pct': (AP_medium * 100.0) if (AP_medium is not None) else None,
                'AP_large_pct': (AP_large * 100.0) if (AP_large is not None) else None,

                'AR_1_pct': (AR_1 * 100.0) if (AR_1 is not None) else None,
                'AR_10_pct': (AR_10 * 100.0) if (AR_10 is not None) else None,
                'AR_100_pct': (AR_100 * 100.0) if (AR_100 is not None) else None,
                'AR_small_pct': (AR_small * 100.0) if (AR_small is not None) else None,
                'AR_medium_pct': (AR_medium * 100.0) if (AR_medium is not None) else None,
                'AR_large_pct': (AR_large * 100.0) if (AR_large is not None) else None,
            }


            if self.output_dir and dist.is_main_process():
                log_path = self.output_dir / "log.txt"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                print(f"[DetSolver] appended log to {log_path}")


                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 10 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        eval_file = self.output_dir / "eval" / name
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, eval_file)
                        print(f"[DetSolver] saved eval to {eval_file}")


        if self.output_dir and dist.is_main_process() and len(epoch_history) > 0:
            epochs = epoch_history

            plt.figure()
            plt.plot(epochs, train_loss_history, label='Train Loss')
            plt.plot(epochs, val_acc_history, label='Val AP (coco_eval_bbox[0])')
            plt.plot(epochs, val_recall_history, label='Val AR@100 (coco_eval_bbox[8])')
            plt.xlabel('Epoch')
            plt.legend()
            plt.title('Loss / AP / AR vs Epoch')

            curve_path = self.output_dir / "curves_loss_ap_ar.png"
            plt.savefig(curve_path)
            plt.close()
            print(f"[DetSolver] saved curves to {curve_path}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('[DetSolver] Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        self._ensure_output_dir()

        print("[DetSolver] building COCO API for validation dataset ...")
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        print("[DetSolver] COCO API ready.")

        module = self.ema.module if self.ema else self.model
        print("[DetSolver] evaluating (val-only) ...")
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, base_ds, self.device, self.output_dir
        )

        if self.output_dir and dist.is_main_process() and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            eval_file = self.output_dir / "eval.pth"
            torch.save(coco_evaluator.coco_eval["bbox"].eval, eval_file)
            print(f"[DetSolver] saved eval to {eval_file}")
        return
