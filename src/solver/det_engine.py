# by xueqianyue
import math
import sys
from typing import Iterable, Any

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict


def _move_to_device(obj: Any, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_device(v, device) for v in obj]
        return type(obj)(t)
    return obj


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    print_freq_pos: int = 10,
                    **kwargs):

    from src.misc import dist


    print_freq = kwargs.pop('print_freq', print_freq_pos)

    debug = kwargs.get('debug', True)
    debug_steps = kwargs.get('debug_steps', 3)
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    amp_dtype = kwargs.get('amp_dtype', torch.float16)

    _dbg_cnt = 0

    model.train()
    criterion.train()


    metric_logger = MetricLogger(delimiter=" | ")
    metric_logger.add_meter('lr',        SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss',      SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_vfl',  SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_emb',  SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_band',  SmoothedValue(window_size=1, fmt='{value:.4f}'))


    header = f'Epoch: [{epoch}]'


    amp_enabled = (scaler is not None) and (device.type == 'cuda')
    amp_device_type = device.type

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = _move_to_device(samples, device)
        targets = _move_to_device(targets, device)

        if amp_enabled:

            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=True):
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:

            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()


        if ema is not None:
            ema.update(model)


        if debug and (_dbg_cnt < debug_steps) and (
                not dist.is_dist_available_and_initialized() or dist.is_main_process()):
            print(f"\n[DBG][epoch {epoch} step {step}] outputs keys = {list(outputs.keys())}")

            if 'pred_embed' in outputs:
                pe = outputs['pred_embed']
                print(f"[DBG] pred_embed shape = {tuple(pe.shape)}")
                with torch.no_grad():
                    norms = torch.linalg.vector_norm(pe[0], ord=2, dim=-1)
                    print(f"[DBG] pred_embed L2-norm (sample #0) "
                          f"min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
            else:
                print("[DBG][WARN] outputs 中未找到 'pred_embed'，请确认 Decoder 改动已生效。")


            print(f"[DBG] loss keys = {list(loss_dict.keys())}")
            if 'loss_emb' in loss_dict:
                print(f"[DBG] loss_emb = {loss_dict['loss_emb'].item():.6f}")
            else:
                print("[DBG][WARN] loss_dict 中未找到 'loss_emb'，"
                      "请确认 Criterion 改动/配置已生效（SetCriterion.losses 包含 'emb'）。")
            _dbg_cnt += 1


        KEEP_LOSSES = {"loss_vfl", "loss_bbox", "loss_giou", "loss_emb", "loss_band"}

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        log_losses = {k: float(loss_dict_reduced.get(k, 0.0)) for k in KEEP_LOSSES}
        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"],
            loss=float(loss_value),
            **log_losses
        )


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             postprocessors,
             data_loader,
             base_ds,
             device: torch.device,
             output_dir):

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'


    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None


    for samples, targets in metric_logger.log_every(data_loader, 10, header):

        samples = _move_to_device(samples, device)
        targets = _move_to_device(targets, device)


        outputs = model(samples)


        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)


        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()


    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
