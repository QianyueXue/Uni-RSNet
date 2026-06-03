import torch
import torch.utils.data
import torchvision
from torchvision import datapoints
from pycocotools import mask as coco_mask
from src.core import register

from PIL import Image


def _dbg_type(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor[{x.dtype}, {tuple(x.shape)}]"
    if isinstance(x, datapoints.Image):
        return f"tv_tensors.Image[{x.dtype}, {tuple(x.shape)}]"
    if isinstance(x, Image.Image):
        return f"PIL[{x.mode}, {x.size}]"
    return type(x).__name__


def _pil_to_chw_uint8_image(pil_img: Image.Image) -> datapoints.Image:

    assert isinstance(pil_img, Image.Image), f"expect PIL.Image, got {type(pil_img)}"


    if pil_img.mode not in ("L", "RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")

    W, H = pil_img.size
    nchan = {"L": 1, "RGB": 3, "RGBA": 4}[pil_img.mode]


    buf = pil_img.tobytes()

    storage = torch.ByteStorage.from_buffer(buf)
    t = torch.ByteTensor(storage)


    t = t.view(H, W, nchan).permute(2, 0, 1).contiguous()
    return datapoints.Image(t)


@register
# by xueqianyue
class CocoDetection_XQY(torchvision.datasets.CocoDetection):


    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category: bool = False):
        super(CocoDetection_XQY, self).__init__(img_folder, ann_file)

        self.img_folder = img_folder
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)

        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category


        self.index2video = []
        missing_cnt = 0
        str_to_int_cache = {}

        for img_id in self.ids:
            meta = self.coco.imgs[img_id]
            raw_vid = meta.get("video_id", -1)
            vid_int = -1
            try:
                vid_int = int(raw_vid)
            except Exception:
                if isinstance(raw_vid, str):
                    if raw_vid not in str_to_int_cache:
                        str_to_int_cache[raw_vid] = len(str_to_int_cache) + 1
                    vid_int = str_to_int_cache[raw_vid]
            if vid_int == -1:
                missing_cnt += 1
            self.index2video.append(vid_int)

        uniq_vids = len(set(self.index2video))
        print(
            f"[CocoDetection_XQY] index2video built. images={len(self.ids)}, "
            f"unique_videos={uniq_vids}, missing_video_id={missing_cnt}"
        )


        self._verbose = True
        self._verbose_n = 3

    def __getitem__(self, idx):

        if isinstance(idx, str):
            idx = int(idx)
        if self._verbose and idx < self._verbose_n:
            print(f"[DBG] __getitem__ idx = {idx} ({type(idx)})")


        img_pil, anno_list = super(CocoDetection_XQY, self).__getitem__(idx)
        assert isinstance(img_pil, Image.Image), f"img 应为 PIL.Image，实际 {type(img_pil)}"


        image_id = self.ids[idx]
        img_meta = self.coco.imgs[image_id]
        raw_video_id = img_meta.get("video_id", -1)
        try:
            video_id_int = int(raw_video_id)
        except Exception:
            video_id_int = int(self.index2video[idx]) if idx < len(self.index2video) else -1

        if self._verbose and idx < self._verbose_n:
            print(f"[DBG] image_id = {image_id} ({type(image_id)})")
            print(f"[DBG] raw_video_id = {raw_video_id} -> video_id_int = {video_id_int}")


        target = {
            "image_id": image_id,
            "annotations": anno_list,
            "video_id": video_id_int
        }


        img_pil, target = self.prepare(img_pil, target)


        if self._verbose and idx < self._verbose_n:
            print(f"[DBG@pre-transform] img={_dbg_type(img_pil)}")
            for k in sorted(target.keys()):
                v = target[k]
                summary = _dbg_type(v)
                if isinstance(v, torch.Tensor) and v.ndim > 0 and v.numel() > 0:
                    preview = v.reshape(-1)[:4].tolist()
                    summary += f" head={preview}"
                print(f"    - {k}: {summary}")


        img = _pil_to_chw_uint8_image(img_pil)


        H, W = target["size"].tolist()
        if "boxes" in target:
            target["boxes"] = datapoints.BoundingBox(
                target["boxes"],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=(H, W),
            )
        if "masks" in target:
            target["masks"] = datapoints.Mask(target["masks"])


        if self._transforms is not None:
            try:
                img, target = self._transforms(img, target)
            except Exception as e:

                print("[ERR] transforms 期间异常！当前各字段类型：")
                print(f"    img: {_dbg_type(img)}")
                for k in sorted(target.keys()):
                    print(f"    - {k}: {_dbg_type(target[k])}")
                raise


        if self._verbose and idx < self._verbose_n:
            print(f"[DBG@Post-transform] img={_dbg_type(img)}")
            for k in sorted(target.keys()):
                print(f"    - {k}: {_dbg_type(target[k])}")

            n_inst = int(target["boxes"].shape[0]) if "boxes" in target else 0
            vid_u = int(target["video_id"][0].item()) if "video_id" in target and n_inst > 0 else -1
            has_inst = "inst_id" in target
            print(
                f"[CocoDetection_XQY][{idx}] image_id={int(image_id)}, video_id={vid_u}, "
                f"instances={n_inst}, has_inst_id={has_inst}"
            )

        return img, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n {repr(self._transforms)}"
        return s


def convert_coco_poly_to_mask(segmentations, height, width):

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]

        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)

    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


# by xueqianyue
class ConvertCocoPolysToMask(object):

    def __init__(self, return_masks: bool = False, remap_mscoco_category: bool = False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self._warn_once_no_inst = False
        self._seen_print = 0

    def __call__(self, image, target):
        assert isinstance(image, Image.Image), f"image 应为 PIL.Image，实际 {type(image)}"
        W, H = image.size


        image_id_val = int(target["image_id"])
        image_id = torch.tensor([image_id_val], dtype=torch.int64)


        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]


        boxes = [obj.get("bbox", [0, 0, 0, 0]) for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=W)
        boxes[:, 1::2].clamp_(min=0, max=H)


        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [int(obj.get("category_id", -1)) for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)


        if self.return_masks:
            segms = [obj.get("segmentation", []) for obj in anno]
            masks = convert_coco_poly_to_mask(segms, H, W)


        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            if keypoints.numel() > 0:
                keypoints = keypoints.view(keypoints.shape[0], -1, 3)


        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]


        if len(anno) > 0 and ("inst_id" in anno[0]):
            inst_ids = torch.as_tensor([int(obj.get("inst_id", -1)) for obj in anno], dtype=torch.int64)[keep]
        else:
            inst_ids = torch.full((boxes.shape[0],), -1, dtype=torch.int64)
            if not self._warn_once_no_inst:
                print("[Convert] WARNING: 'inst_id' not found; filled with -1.")
                self._warn_once_no_inst = True


        raw_vid = target.get("video_id", -1)
        try:
            vid_int = int(raw_vid)
        except Exception:
            vid_int = -1
        video_id = torch.full((boxes.shape[0],), vid_int, dtype=torch.int64)


        area = torch.tensor([float(obj.get("area", 0.0)) for obj in anno], dtype=torch.float32)[keep]
        iscrowd = torch.tensor([int(obj["iscrowd"]) if "iscrowd" in obj else 0 for obj in anno],
                               dtype=torch.int64)[keep]

        out = {
            "boxes": boxes,
            "labels": classes,
            "inst_id": inst_ids,
            "video_id": video_id,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([H, W], dtype=torch.int64),
            "size": torch.as_tensor([H, W], dtype=torch.int64),
        }

        if self.return_masks:
            out["masks"] = masks
        if keypoints is not None:
            out["keypoints"] = keypoints


        if boxes.numel() > 0 and self._seen_print < 2:
            uniq_inst = int(inst_ids.unique().numel())
            print(f"[Convert] image={image_id_val}, video={vid_int}, instances={int(boxes.shape[0])}, uniq_inst_id={uniq_inst}")
            self._seen_print += 1

        return image, out


mscoco_category2name = {
    0: "lfm",
    1: "sfm",
    2: "bpsk",
    3: "fsk",
    4: "COSTAS",
    5: "eqfm",
    6: "nlfm",
    7: "fmcw",
}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
