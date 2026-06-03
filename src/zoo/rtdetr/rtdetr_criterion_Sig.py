# by xueqianyue

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


@register
# by xueqianyue
class SetCriterion_XQY(nn.Module):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 eos_coef=1e-4,
                 num_classes=80,

                 emb_tau=0.07,
                 min_pos=1,
                 emb_mem_enable=False,
                 emb_mem_size=32,
                 emb_mem_neg=256,

                 band_use_logh=False,
                 band_eps=1e-6,
                 band_min_group=2,
                 band_proto_enable=True,
                 band_proto_m=0.9,
                 band_within_w=1.0,
                 band_proto_w=1.0,

                 debug=True,
                 debug_steps=3):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.losses = list(losses)


        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)


        self.alpha = alpha
        self.gamma = gamma


        self.emb_tau = float(emb_tau)
        self.min_pos = int(min_pos)
        self.emb_mem_enable = bool(emb_mem_enable)
        self.emb_mem_size = int(emb_mem_size)
        self.emb_mem_neg = int(emb_mem_neg)

        self._emb_memory = {}


        self.band_use_logh = bool(band_use_logh)
        self.band_eps = float(band_eps)
        self.band_min_group = int(band_min_group)
        self.band_proto_enable = bool(band_proto_enable)
        self.band_proto_m = float(band_proto_m)
        self.band_within_w = float(band_within_w)
        self.band_proto_w = float(band_proto_w)

        self._band_proto = {}


        self.debug = bool(debug)
        self.debug_steps = int(debug_steps)
        self._dbg_cnt = 0
        self._printed_skip = set()


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}


    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}


    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs and 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_onehot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target_onehot

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target_onehot) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_embed_supcon(self, outputs, targets, indices, num_boxes, log=True):

        dev = outputs['pred_logits'].device if 'pred_logits' in outputs else None


        if ('pred_embed' not in outputs) or (outputs['pred_embed'] is None):
            self._maybe_print_once("[Criterion][SKIP] emb not executed: 'pred_embed' missing.")
            return {'loss_emb': torch.zeros([], device=dev, dtype=torch.float32)}

        z_all = outputs['pred_embed']
        B, Nq, D = z_all.shape
        idx = self._get_src_permutation_idx(indices)
        z = z_all[idx]
        if z.numel() == 0:
            self._maybe_print_once("[Criterion][SKIP] emb not executed: no matched pairs (M=0).")
            return {'loss_emb': z.sum() * 0.0}


        z = torch.nn.functional.normalize(z, dim=-1)


        gt_inst = torch.cat([t['inst_id'][J] for t, (_, J) in zip(targets, indices)], dim=0).to(z.device)
        gt_vid  = torch.cat([t['video_id'][J] for t, (_, J) in zip(targets, indices)], dim=0).to(z.device)

        tau = max(float(self.emb_tau), 1e-6)
        sim = (z @ z.t()) / tau
        I = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        same_video = (gt_vid.unsqueeze(1) == gt_vid.unsqueeze(0))
        pos_mask = same_video & (gt_inst.unsqueeze(1) == gt_inst.unsqueeze(0)) & (~I)
        valid_cols = (~I) & same_video
        has_valid = valid_cols.any(dim=1, keepdim=True)

        sim_masked = sim.masked_fill(~valid_cols, float('-inf'))
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
        log_denom = torch.where(has_valid, log_denom, torch.zeros_like(log_denom))
        log_prob = sim_masked - log_denom
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

        pos_count = pos_mask.sum(dim=1).clamp(min=int(self.min_pos))
        loss_i = -(log_prob * pos_mask).sum(dim=1) / pos_count
        loss = loss_i.mean()


        if self.emb_mem_enable:

            no_pos_mask = (pos_mask.sum(dim=1) == 0)
            if no_pos_mask.any():


                loss_fallback = []
                for i in torch.nonzero(no_pos_mask, as_tuple=False).flatten().tolist():
                    key = (int(gt_vid[i].item()), int(gt_inst[i].item()))
                    z_i = z[i:i+1]
                    pos_list = self._emb_memory.get(key, [])
                    if len(pos_list) == 0:
                        continue
                    pos = torch.stack(pos_list, dim=0).to(z.device)

                    neg_samples = []
                    if self.emb_mem_neg > 0:
                        for k2, lst in self._emb_memory.items():
                            if k2 != key and len(lst) > 0:
                                neg_samples.extend(lst)
                        if len(neg_samples) > 0:
                            rng_idx = torch.randperm(len(neg_samples))[:self.emb_mem_neg]
                            neg = torch.stack([neg_samples[j] for j in rng_idx.tolist()], dim=0).to(z.device)
                        else:
                            neg = None
                    else:
                        neg = None


                    s_pos = (z_i @ pos.t()) / tau
                    num = torch.logsumexp(s_pos, dim=1)
                    if neg is not None and neg.numel() > 0:
                        s_neg = (z_i @ neg.t()) / tau
                        den = torch.logsumexp(torch.cat([s_pos, s_neg], dim=1), dim=1)
                    else:
                        den = num
                    loss_fallback.append(-(num - den))
                if len(loss_fallback) > 0:
                    loss_fallback = torch.cat(loss_fallback).mean()

                    loss = 0.5 * loss + 0.5 * loss_fallback


        if self.emb_mem_enable:
            with torch.no_grad():
                for i in range(z.shape[0]):
                    key = (int(gt_vid[i].item()), int(gt_inst[i].item()))
                    if key[1] < 0:
                        continue
                    lst = self._emb_memory.get(key, [])
                    lst.append(z[i].detach())
                    if len(lst) > self.emb_mem_size:
                        lst = lst[-self.emb_mem_size:]
                    self._emb_memory[key] = lst


        self._maybe_print_once(f"[Criterion][DO] emb executed: M={z.shape[0]}, tau={tau:.4f}, loss_emb={float(loss):.6f}")
        return {'loss_emb': loss}


    def loss_band(self, outputs, targets, indices, num_boxes, log=True):

        if 'pred_boxes' not in outputs:
            self._maybe_print_once("[Criterion][SKIP] band not executed: 'pred_boxes' missing.")
            dev = outputs['pred_logits'].device if 'pred_logits' in outputs else None
            return {'loss_band': torch.zeros([], device=dev, dtype=torch.float32)}

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if src_boxes.numel() == 0:
            self._maybe_print_once("[Criterion][SKIP] band not executed: no matched pairs (M=0).")
            return {'loss_band': src_boxes.sum() * 0.0}

        cy = src_boxes[:, 1]
        w  = src_boxes[:, 2]
        h  = src_boxes[:, 3]


        gt_inst = torch.cat([t['inst_id'][J] for t, (_, J) in zip(targets, indices)], dim=0)
        gt_vid  = torch.cat([t['video_id'][J] for t, (_, J) in zip(targets, indices)], dim=0)


        keys = torch.stack([gt_vid.to(torch.int64), gt_inst.to(torch.int64)], dim=1)
        uniq_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
        group_losses = []
        for g in range(uniq_keys.shape[0]):
            mask = (inv == g)
            cnt = int(mask.sum().item())
            if cnt >= self.band_min_group and (uniq_keys[g, 1].item() >= 0):
                cy_g = cy[mask]
                w_g  = w[mask]

                loss_cy = ((cy_g - cy_g.mean()) ** 2).mean()

                loss_w  = ((w_g  - w_g.mean())  ** 2).mean()

                if self.band_use_logh:
                    h_g = h[mask].clamp(min=self.band_eps)
                    logh_g = torch.log(h_g)
                    loss_logh = ((logh_g - logh_g.mean()) ** 2).mean()
                else:
                    loss_logh = cy_g.sum() * 0.0
                group_losses.append(loss_cy + loss_w + loss_logh)

        loss_within = torch.stack(group_losses).mean() if len(group_losses) > 0 else cy.sum() * 0.0


        loss_proto = cy.sum() * 0.0
        if self.band_proto_enable:


            proto_losses = []
            for g in range(uniq_keys.shape[0]):
                mask = (inv == g)
                cnt = int(mask.sum().item())
                vid_i, inst_i = int(uniq_keys[g, 0].item()), int(uniq_keys[g, 1].item())
                if inst_i < 0:
                    continue
                cy_g = cy[mask]; w_g = w[mask]; h_g = h[mask]
                cy_mean = cy_g.mean().detach()
                w_mean  = w_g.mean().detach()
                h_mean  = h_g.mean().detach()

                key = (vid_i, inst_i)
                if key not in self._band_proto:
                    self._band_proto[key] = {
                        'cy': cy_mean.clone(),
                        'w' : w_mean.clone(),
                        'h' : h_mean.clone(),
                    }
                else:

                    self._band_proto[key]['cy'] = self.band_proto_m * self._band_proto[key]['cy'] + (1 - self.band_proto_m) * cy_mean
                    self._band_proto[key]['w']  = self.band_proto_m * self._band_proto[key]['w']  + (1 - self.band_proto_m) * w_mean
                    self._band_proto[key]['h']  = self.band_proto_m * self._band_proto[key]['h']  + (1 - self.band_proto_m) * h_mean


                proto = self._band_proto[key]
                loss_cy_p = ((cy_g - proto['cy']) ** 2).mean()
                loss_w_p  = ((w_g  - proto['w'])  ** 2).mean()
                if self.band_use_logh:
                    loss_logh_p = ((torch.log(h_g.clamp(min=self.band_eps)) - torch.log(proto['h'].clamp(min=self.band_eps))) ** 2).mean()
                else:
                    loss_logh_p = cy_g.sum() * 0.0

                proto_losses.append(loss_cy_p + loss_w_p + loss_logh_p)

            loss_proto = torch.stack(proto_losses).mean() if len(proto_losses) > 0 else loss_proto


        loss_band = self.band_within_w * loss_within + self.band_proto_w * loss_proto
        self._maybe_print_once(f"[Criterion][DO] band executed: groups_within={len(group_losses)}, "
                               f"within={float(loss_within):.6f}, proto={float(loss_proto):.6f}, "
                               f"loss_band={float(loss_band):.6f}")
        return {'loss_band': loss_band}


    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,

            'emb': self.loss_embed_supcon,
            'band': self.loss_band,

        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):


        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}


        indices = self.matcher(outputs_without_aux, targets)


        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()


        losses = {}
        for loss in self.losses:

            if loss == 'boxes':
                if self.weight_dict.get('loss_bbox', 0.0) == 0.0 and self.weight_dict.get('loss_giou', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] boxes not executed: weight=0 (loss_bbox & loss_giou).")
                    continue
            elif loss == 'labels':
                if self.weight_dict.get('loss_ce', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] labels not executed: weight=0 (loss_ce).")
                    continue
            elif loss == 'bce':
                if self.weight_dict.get('loss_bce', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] bce not executed: weight=0 (loss_bce).")
                    continue
            elif loss == 'focal':
                if self.weight_dict.get('loss_focal', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] focal not executed: weight=0 (loss_focal).")
                    continue
            elif loss == 'vfl':
                if self.weight_dict.get('loss_vfl', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] vfl not executed: weight=0 (loss_vfl).")
                    continue
            elif loss == 'emb':
                if self.weight_dict.get('loss_emb', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] emb not executed: weight=0 (loss_emb).")
                    continue
            elif loss == 'band':
                if self.weight_dict.get('loss_band', 0.0) == 0.0:
                    self._maybe_print_once("[Criterion][SKIP] band not executed: weight=0 (loss_band).")
                    continue


            l_dict = self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes)

            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)


        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_i = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ('emb', 'band', 'masks'):

                        continue

                    if loss == 'boxes':
                        if self.weight_dict.get('loss_bbox', 0.0) == 0.0 and self.weight_dict.get('loss_giou', 0.0) == 0.0:
                            continue
                    elif loss == 'labels' and self.weight_dict.get('loss_ce', 0.0) == 0.0:
                        continue
                    elif loss == 'bce' and self.weight_dict.get('loss_bce', 0.0) == 0.0:
                        continue
                    elif loss == 'focal' and self.weight_dict.get('loss_focal', 0.0) == 0.0:
                        continue
                    elif loss == 'vfl' and self.weight_dict.get('loss_vfl', 0.0) == 0.0:
                        continue

                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, 'dn_meta is required when dn_aux_outputs present'
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss in ('emb', 'band', 'masks'):
                        continue
                    if loss == 'boxes':
                        if self.weight_dict.get('loss_bbox', 0.0) == 0.0 and self.weight_dict.get('loss_giou', 0.0) == 0.0:
                            continue
                    elif loss == 'labels' and self.weight_dict.get('loss_ce', 0.0) == 0.0:
                        continue
                    elif loss == 'bce' and self.weight_dict.get('loss_bce', 0.0) == 0.0:
                        continue
                    elif loss == 'focal' and self.weight_dict.get('loss_focal', 0.0) == 0.0:
                        continue
                    elif loss == 'vfl' and self.weight_dict.get('loss_vfl', 0.0) == 0.0:
                        continue

                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                                         torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices


    def _maybe_print_once(self, msg: str):
        if not self.debug:
            return
        if self._dbg_cnt >= self.debug_steps:
            return
        if msg in self._printed_skip:
            return

        if (not is_dist_available_and_initialized()) or torch.distributed.get_rank() == 0:
            print(msg)
        self._printed_skip.add(msg)
        self._dbg_cnt += 1


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
