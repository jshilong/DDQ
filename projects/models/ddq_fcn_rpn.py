import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.models import HEADS
from mmdet.models.dense_heads.paa_head import levels_to_images

from .ddq_fcn_head import DDQFCNHead
from .utils import align_tensor


@HEADS.register_module()
class DDQFCNRPN(DDQFCNHead):
    def __init__(self, *args, num_distinct_queries=300, **kwargs):
        self.num_distinct_queries = num_distinct_queries

        super(DDQFCNRPN, self).__init__(*args, **kwargs)

    def _init_layers(self):
        super(DDQFCNRPN, self)._init_layers()
        self.compress = nn.Linear(self.feat_channels * 2, self.feat_channels)

    def get_inputs(self, main_results, aux_results, img_metas=None):

        mlvl_score = main_results['cls_scores_list']

        num_levels = len(mlvl_score)
        featmap_sizes = [mlvl_score[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=mlvl_score[0].dtype,
            device=mlvl_score[0].device)

        all_cls_scores, all_bbox_preds, all_query_ids = self.pre_dqs(
            **main_results, mlvl_priors=mlvl_priors, img_metas=img_metas)
        if aux_results is None:
            (aux_cls_scores, aux_bbox_preds) = (all_cls_scores, all_bbox_preds)
        else:
            aux_cls_scores, aux_bbox_preds, aux_query_ids = self.pre_dqs(
                **aux_results, mlvl_priors=mlvl_priors, img_metas=img_metas)

        dqs_all_cls_scores, dqs_all_bbox_preds, dqs_query_ids = self.dqs(
            all_cls_scores, all_bbox_preds, all_query_ids)

        distinct_query_dict = self.construct_query(main_results,
                                                   dqs_all_cls_scores,
                                                   dqs_all_bbox_preds,
                                                   dqs_query_ids,
                                                   self.num_distinct_queries)


        return (dqs_all_cls_scores, dqs_all_bbox_preds), \
                (aux_cls_scores, aux_bbox_preds), \
                    distinct_query_dict

    def construct_query(self, feat_dict, scores, bboxes, query_ids, num=300):
        cls_feats = feat_dict['cls_feats']
        reg_feats = feat_dict['reg_feats']
        cls_feats = levels_to_images(cls_feats)
        reg_feats = levels_to_images(reg_feats)
        num_img = len(cls_feats)
        all_img_proposals = []
        all_img_object_feats = []
        for img_id in range(num_img):
            singl_scores = scores[img_id].max(-1).values
            singl_bboxes = bboxes[img_id]
            single_ids = query_ids[img_id]
            singl_cls_feats = cls_feats[img_id]
            singl_reg_feats = reg_feats[img_id]

            object_feats = torch.cat([singl_cls_feats, singl_reg_feats],
                                     dim=-1)

            object_feats = object_feats.detach()
            singl_bboxes = singl_bboxes.detach()

            object_feats = self.compress(object_feats)

            select_ids = torch.sort(singl_scores,
                                    descending=True).indices[:num]
            single_ids = single_ids[select_ids]
            singl_bboxes = singl_bboxes[select_ids]

            object_feats = object_feats[single_ids]
            all_img_object_feats.append(object_feats)
            all_img_proposals.append(singl_bboxes)

        all_img_object_feats = align_tensor(all_img_object_feats)
        all_img_proposals = align_tensor(all_img_proposals)
        return dict(proposals=all_img_proposals,
                    object_feats=all_img_object_feats)

    def simple_test_rpn(self, x, img_metas, **kwargs):

        loss = dict()

        main_results, aux_results = self.forward(x)
        main_loss_inputs, aux_loss_inputs, \
            distinc_query_dict = self.get_inputs(
            main_results, aux_results, img_metas=img_metas)

        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        return loss, imgs_whwh, distinc_query_dict

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        self.img_metas = img_metas
        self.gt_bboxes = gt_bboxes
        loss = dict()

        main_results, aux_results = self.forward(x)
        main_loss_inputs, aux_loss_inputs, \
            distinc_query_dict = self.get_inputs(
            main_results, aux_results, img_metas=img_metas)

        aux_loss = self.aux_loss(*aux_loss_inputs,
                                 gt_bboxes=gt_bboxes,
                                 gt_labels=gt_labels,
                                 img_metas=img_metas)
        for k, v in aux_loss.items():
            loss[f'aux_{k}'] = v

        main_loss = self.main_loss(*main_loss_inputs,
                                   gt_bboxes=gt_bboxes,
                                   gt_labels=gt_labels,
                                   img_metas=img_metas)

        loss.update(main_loss)

        loss['num_proposal'] = torch.as_tensor(
            sum([len(item) for item in main_loss_inputs[0]
                 ])).cuda().float() / len(main_loss_inputs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        return loss, imgs_whwh, distinc_query_dict

    def simple_test(self, x, img_metas, **kwargs):

        main_results, aux_results = self.forward(x)
        main_outs, _ = self.get_inputs(main_results,
                                       aux_results,
                                       img_metas=img_metas)

        results_list = self.get_bboxes(*main_outs, img_metas)
        return results_list

    def dqs(self, all_mlvl_scores, all_mlvl_bboxes, all_query_ids, **kwargs):
        all_distinct_bboxes = []
        all_distinct_scores = []
        all_distinct_query_ids = []
        for mlvl_bboxes, mlvl_scores, query_id in zip(all_mlvl_bboxes,
                                                      all_mlvl_scores,
                                                      all_query_ids):
            if mlvl_bboxes.numel() == 0:
                return mlvl_bboxes, mlvl_scores, query_id

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes,
                                                mlvl_scores.max(-1).values,
                                                torch.ones(len(mlvl_scores)),
                                                self.dqs_cfg)

            all_distinct_bboxes.append(mlvl_bboxes[keep_idxs])
            all_distinct_scores.append(mlvl_scores[keep_idxs])
            all_distinct_query_ids.append(query_id[keep_idxs])
        return all_distinct_scores, all_distinct_bboxes, all_distinct_query_ids
