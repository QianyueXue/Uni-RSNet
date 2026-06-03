
# by xueqianyue
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor_XQY']


@register
# by xueqianyue
class RTDETRPostProcessor_XQY(nn.Module):

    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=4, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False,label_map=None) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.label_map = label_map
    def extra_repr(self) -> str:

        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'


    def forward(self, outputs, orig_target_sizes):

        from src.misc import dist

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        has_embed = 'pred_embed' in outputs
        if has_embed:
            embeds = outputs['pred_embed']


        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores_all = torch.sigmoid(logits)
            scores, index = torch.topk(scores_all.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index_q = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index_q.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            if has_embed:
                embeds = embeds.gather(dim=1, index=index_q.unsqueeze(-1).repeat(1, 1, embeds.shape[-1]))
        else:
            scores_all = torch.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores_all.max(dim=-1)
            boxes = bbox_pred
            if boxes.shape[1] > self.num_top_queries:
                scores, index_q = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index_q)
                boxes = torch.gather(boxes, dim=1, index=index_q.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
                if has_embed:
                    embeds = torch.gather(embeds, dim=1, index=index_q.unsqueeze(-1).repeat(1, 1, embeds.shape[-1]))


        if self.deploy_mode:
            if has_embed:
                return labels, boxes, scores, embeds
            return labels, boxes, scores


        if self.remap_mscoco_category and self.label_map is not None:
            labels = torch.tensor(
                [self.label_map[int(x.item())] for x in labels.flatten()]
            ).to(boxes.device).reshape(labels.shape)

        results = []
        for b in range(labels.shape[0]):
            item = dict(labels=labels[b], boxes=boxes[b], scores=scores[b])
            if has_embed:
                item['embeddings'] = embeds[b]
            results.append(item)


        if not hasattr(self, '_dbg_cnt'):
            self._dbg_cnt = 0
        if self._dbg_cnt < 2 and (not dist.is_dist_available_and_initialized() or dist.is_main_process()):
            print(f"[PostProcessor][DBG] use_focal={self.use_focal_loss}, num_top={self.num_top_queries}, "
                  f"has_embed={has_embed}")
            print(f"[PostProcessor][DBG] sample #0 -> K={labels.shape[1]}, "
                  f"top5 scores={results[0]['scores'][:5].detach().cpu().numpy().round(4)}")
            if has_embed:
                with torch.no_grad():
                    z0 = results[0]['embeddings']
                    norms = torch.linalg.vector_norm(z0, ord=2, dim=-1)
                    print(f"[PostProcessor][DBG] embeddings shape={tuple(z0.shape)}, "
                          f"L2-norm(min/mean/max)={norms.min().item():.4f}/"
                          f"{norms.mean().item():.4f}/{norms.max().item():.4f}")
            self._dbg_cnt += 1

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self, ):
        return ('bbox', )


from src.core import register
