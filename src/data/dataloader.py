import importlib
from typing import Any, Mapping, Sequence, Optional

import torch
from torch.utils import data

from src.core import register


def _resolve_class(typ: Any):

    if isinstance(typ, type):
        return typ
    if isinstance(typ, str):
            if "." in typ:
                module_name, cls_name = typ.rsplit(".", 1)
                return getattr(importlib.import_module(module_name), cls_name)
            if typ == "SigBatchSampler":
                from .samplers import SigBatchSampler
                return SigBatchSampler
    raise ValueError(f"无法解析类型：{typ!r}")


@register
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __init__(
        self,
        dataset,
        batch_size: Optional[int] = None,
        sampler=None,
        batch_sampler=None,
        shuffle: Optional[bool] = None,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        **kwargs,
    ) -> None:

        using_batch_sampler = False


        if isinstance(batch_sampler, dict):
            cfg = dict(batch_sampler)
            typ = cfg.pop("type", None)
            SamplerClass = _resolve_class(typ)
            batch_sampler = SamplerClass(dataset, **cfg)
            using_batch_sampler = True


        elif batch_sampler is not None:
            using_batch_sampler = True


        if using_batch_sampler:
            assert sampler is None, "使用 batch_sampler 时请勿再传 sampler。"
            assert batch_size is None, "使用 batch_sampler 时请勿再传 batch_size。"
            assert shuffle in (None, False), "使用 batch_sampler 时请勿再传 shuffle=True。"
            assert drop_last in (None, False), "使用 batch_sampler 时请勿再传 drop_last。"

            super().__init__(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                **kwargs,
            )
        else:
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=shuffle if shuffle is not None else False,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=collate_fn,
                **kwargs,
            )

    def __repr__(self) -> str:
        parts = [self.__class__.__name__ + "("]
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            parts.append(f"    {n}: {getattr(self, n, None)}")
        parts.append(f"    sampler: {getattr(self, 'sampler', None)}")
        bs = getattr(self, "batch_sampler", None)
        parts.append(f"    batch_sampler: {type(bs) if bs is not None else None}")
        parts.append(")")
        return "\n".join(parts)


@register
# by xueqianyue
def default_collate_fn(items):

    from typing import Mapping, Sequence
    assert isinstance(items, Sequence) and len(items) > 0, "batch 为空？检查 batch_sampler / dataset"

    def split_item(it):
        if isinstance(it, Mapping):
            if "image" in it:
                img = it["image"]
                tgt = {k: v for k, v in it.items() if k != "image"}
                return img, tgt
            if "img" in it:
                img = it["img"]
                tgt = {k: v for k, v in it.items() if k != "img"}
                return img, tgt
            raise KeyError("字典样本缺少 'image'/'img' 键")
        else:
            assert isinstance(it, Sequence) and len(it) >= 2, "样本应为 (image, target)"
            return it[0], it[1]

    images, targets = zip(*[split_item(it) for it in items])

    imgs = []
    for im in images:
        if not isinstance(im, torch.Tensor):
            im = torch.as_tensor(im)
        if im.ndim == 2:
            im = im.unsqueeze(0)
        elif im.ndim == 3 and im.shape[-1] in (1, 3, 4):
            im = im.permute(2, 0, 1).contiguous()
        assert im.ndim == 3, f"image 维度异常: {im.shape}"
        imgs.append(im)

    shapes = {tuple(x.shape) for x in imgs}
    assert len(shapes) == 1, f"同一 batch 图像尺寸不一致: {shapes}，请在 transforms 中统一 Resize/Pad"
    images = torch.stack(imgs, dim=0)

    out_targets = []
    for t in targets:
        if not isinstance(t, Mapping):
            raise TypeError("target 应为 dict")
        t_out = {}

        boxes = t.get("boxes", None)
        if boxes is None:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes)
            if boxes_t.numel() == 0:
                boxes_t = boxes_t.reshape(0, 4)
            assert boxes_t.shape[-1] == 4, f"boxes 最后维必须=4, got {boxes_t.shape}"
            boxes_t = boxes_t.to(dtype=torch.float32)
        t_out["boxes"] = boxes_t

        labels = t.get("labels", None)
        if labels is None:
            labels_t = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
        else:
            labels_t = torch.as_tensor(labels)
            if labels_t.numel() == 0:
                labels_t = labels_t.reshape(0)
            labels_t = labels_t.to(dtype=torch.int64)
            assert labels_t.shape[0] == boxes_t.shape[0], \
                f"labels 与 boxes 数量不一致: {labels_t.shape[0]} vs {boxes_t.shape[0]}"
        t_out["labels"] = labels_t

        def _maybe_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype) if dtype is not None else x
            try:
                tx = torch.as_tensor(x)
                if dtype is not None:
                    tx = tx.to(dtype=dtype)
                return tx
            except Exception:
                return x

        passthrough_keys = [
            "image_id", "area", "iscrowd", "size", "orig_size",
            "inst_id", "video_id", "ids", "keep", "mask", "masks",
        ]
        for k in passthrough_keys:
            if k in t:
                v = t[k]
                if k in ("image_id", "inst_id", "video_id"):
                    t_out[k] = _maybe_tensor(v, dtype=torch.int64)
                elif k in ("size", "orig_size"):
                    vv = _maybe_tensor(v, dtype=torch.int64)
                    vv = vv.reshape(2) if vv.numel() == 2 else vv
                    t_out[k] = vv
                elif k in ("mask", "masks"):
                    t_out[k] = _maybe_tensor(v)
                else:
                    t_out[k] = _maybe_tensor(v)

        for k, v in t.items():
            if k not in t_out:
                t_out[k] = v

        out_targets.append(t_out)

    return images, out_targets
