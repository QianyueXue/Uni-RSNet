

import torch
import torch.nn as nn

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose_XQY']


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)


@register
# by xueqianyue
class Compose_XQY(T.Compose):


    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):

                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError('Compose：未知的 transform 配置类型')
        else:
            transforms = [EmptyTransform(), ]
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):


    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):

    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:

        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]

        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        self.spatial_size = spatial_size

        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)

        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
# by xueqianyue
class RandomIoUCrop(T.RandomIoUCrop):

    def __init__(self, min_scale: float = 0.3, max_scale: float = 1,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2,
                 sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:

        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        outputs = super().forward(*inputs)


        if isinstance(outputs, (tuple, list)) and len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['__need_sync_extra__'] = True

        return outputs


@register
# by xueqianyue
class SanitizeBoundingBox(T.Transform):

    _transformed_types = (
        datapoints.BoundingBox,
    )

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = float(min_size)
        self._warned_mismatch_once = False

    def _build_keep_mask(self, boxes: datapoints.BoundingBox) -> torch.Tensor:

        fmt = boxes.format

        if fmt != datapoints.BoundingBoxFormat.XYXY:
            xyxy = torchvision.ops.box_convert(
                boxes,
                in_fmt=fmt.value.lower(),
                out_fmt='xyxy'
            )
        else:
            xyxy = boxes


        x1, y1, x2, y2 = xyxy.unbind(dim=-1)
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)

        keep = (w >= self.min_size) & (h >= self.min_size)
        return keep

    def _apply_keep(self, target: Dict[str, Any], keep: torch.Tensor) -> None:

        K = int(keep.sum().item())


        if 'labels' in target and isinstance(target['labels'], torch.Tensor):
            target['labels'] = target['labels'][keep]
        if 'masks' in target and isinstance(target['masks'], torch.Tensor):
            target['masks'] = target['masks'][keep]
        if 'keypoints' in target and isinstance(target['keypoints'], torch.Tensor):
            target['keypoints'] = target['keypoints'][keep]


        for k in ('inst_id', 'video_id'):
            if k in target and isinstance(target[k], torch.Tensor):
                if target[k].ndim == 1:

                    if target[k].shape[0] != keep.shape[0]:

                        if not self._warned_mismatch_once:
                            print(f"[SanitizeBBox][WARN] 字段 '{k}' 长度({target[k].shape[0]}) "
                                  f"与 boxes 长度({keep.shape[0]}) 不一致，已尝试按最小长度对齐。"
                                  f"（多见于 RandomIoUCrop 后未同步自定义字段）")
                            self._warned_mismatch_once = True
                        min_n = min(target[k].shape[0], keep.shape[0])
                        target[k] = target[k][:min_n]

                        keep = keep[:min_n]
                    target[k] = target[k][keep]
                else:

                    if not self._warned_mismatch_once:
                        print(f"[SanitizeBBox][WARN] 字段 '{k}' 形状 {tuple(target[k].shape)} 未识别，未同步 keep。")
                        self._warned_mismatch_once = True


    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        return inpt

    def __call__(self, *inputs: Any) -> Any:

        outputs = T.Transform.__call__(self, *inputs)


        if isinstance(outputs, (list, tuple)) and len(outputs) > 1 and isinstance(outputs[1], dict):
            img, target = outputs[0], outputs[1]
        else:
            return outputs


        if 'boxes' not in target:
            return outputs

        boxes = target['boxes']
        if not isinstance(boxes, datapoints.BoundingBox):

            return outputs


        keep = self._build_keep_mask(boxes)


        if target.pop('__need_sync_extra__', False):

            self._apply_keep(target, keep)
        else:

            self._apply_keep(target, keep)

        return outputs


@register
# by xueqianyue
class ConvertBox(T.Transform):

    _transformed_types = (
        datapoints.BoundingBox,
    )

    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)


        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


from src.core import register
