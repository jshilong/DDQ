import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmdet.core import (MlvlPointGenerator, distance2bbox,
                        filter_scores_and_topk, select_single_mlvl)
from mmdet.models import HEADS, AnchorFreeHead
from mmdet.models.utils import sigmoid_geometric_mean

from .loss_module import AuxLoss


@HEADS.register_module()
class DDQFCNHead(AnchorFreeHead):
    def __init__(
            self,
            *args,
            strides=(8, 16, 32, 64, 128),
            aux_loss=ConfigDict(
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    activated=True,  # use probability instead of logit as input
                    beta=2.0,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                train_cfg=dict(assigner=dict(type='TopkHungarianAssigner',
                                             topk=8),
                               alpha=1,
                               beta=6),
            ),
            main_loss=ConfigDict(
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    activated=True,  # use probability instead of logit as input
                    beta=2.0,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                train_cfg=dict(assigner=dict(type='TopkHungarianAssigner',
                                             topk=1),
                               alpha=1,
                               beta=6),
            ),
            shuffle_channles=64,
            dqs_cfg=dict(type='nms', iou_threshold=0.7, nms_pre=1000),
            offset=0.5,
            num_distinct_queries=300,
            **kwargs):
        self.num_distinct_queries = num_distinct_queries
        self.dqs_cfg = dqs_cfg
        super(DDQFCNHead, self).__init__(*args, strides=strides, **kwargs)
        self.aux_loss = AuxLoss(**aux_loss)
        self.main_loss = AuxLoss(**main_loss)

        self.shuffle_channles = shuffle_channles

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = len(self.prior_generator.strides)
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.in_channels - self.shuffle_channles * 2
        self.init_weights()
        self.prior_generator = MlvlPointGenerator(strides, offset=offset)

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for layer in self.objectness.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

        # only be used in training
        normal_init(self.aux_conv_objectness[-1], std=0.01, bias=bias_cls)
        normal_init(self.aux_conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.aux_conv_reg, std=0.01)

    def get_targets(self):
        pass

    def loss(self, ):
        pass

    def _init_layers(self):
        self.inter_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))

        self.objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2))

        cls_out_channels = self.num_classes

        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * cls_out_channels,
                                  3,
                                  padding=3 // 2)

        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * 4,
                                  3,
                                  padding=3 // 2)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.aux_conv_objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2))

        cls_out_channels = self.num_classes

        self.aux_conv_cls = nn.Conv2d(self.feat_channels,
                                      self.num_base_priors * cls_out_channels,
                                      3,
                                      padding=3 // 2)

        self.aux_conv_reg = nn.Conv2d(self.feat_channels,
                                      self.num_base_priors * 4,
                                      3,
                                      padding=3 // 2)
        self.aux_scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:,
                                     self.remain_chs:][:,
                                                       self.shuffle_channles:]
                from_top = F.interpolate(from_top,
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)
                from_down = down_input[:, self.remain_chs:][:, :self.
                                                            shuffle_channles]
                from_down = F.interpolate(from_down,
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True)
                fused_inputs.append(
                    torch.cat([remain, from_top, from_down], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def forward(self, inputs, **kwargs):
        cls_convs = self.cls_convs
        reg_convs = self.reg_convs
        scales = self.scales
        conv_objectness = self.objectness
        conv_cls = self.conv_cls
        conv_reg = self.conv_reg
        cls_feats = inputs
        reg_feats = inputs

        cls_scores_list = []
        bbox_preds_list = []

        for layer_index, conv_m in enumerate(cls_convs):
            # shuffle last 2 feature maps
            if layer_index > 1:
                cls_feats = self._single_shuffle(cls_feats, [conv_m])
            else:
                cls_feats = [conv_m(item) for item in cls_feats]

        for layer_index, conv_m in enumerate(reg_convs):
            # shuffle last feature maps
            if layer_index > 2:
                reg_feats = self._single_shuffle(reg_feats, [conv_m])
            else:
                reg_feats = [conv_m(item) for item in reg_feats]
        for idx, (cls_feat, reg_feat,
                  scale) in enumerate(zip(cls_feats, reg_feats, scales)):
            cls_logits = conv_cls(cls_feat)
            object_nesss = conv_objectness(reg_feat)
            cls_score = sigmoid_geometric_mean(cls_logits, object_nesss)
            reg_dist = scale(conv_reg(reg_feat).exp()).float()
            cls_scores_list.append(cls_score)
            bbox_preds_list.append(reg_dist)

        main_results = dict(cls_scores_list=cls_scores_list,
                            bbox_preds_list=bbox_preds_list,
                            cls_feats=cls_feats,
                            reg_feats=reg_feats)
        if self.training:
            cls_scores_list = []
            bbox_preds_list = []

            for idx, (cls_feat, reg_feat, scale) in enumerate(
                    zip(cls_feats, reg_feats, self.aux_scales)):
                cls_logits = self.aux_conv_cls(cls_feat)
                object_nesss = self.aux_conv_objectness(reg_feat)
                cls_score = sigmoid_geometric_mean(cls_logits, object_nesss)
                reg_dist = scale(self.aux_conv_reg(reg_feat).exp()).float()
                cls_scores_list.append(cls_score)
                bbox_preds_list.append(reg_dist)
            aux_results = dict(cls_scores_list=cls_scores_list,
                               bbox_preds_list=bbox_preds_list,
                               cls_feats=cls_feats,
                               reg_feats=reg_feats)
        else:
            aux_results = None
        return main_results, aux_results

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        loss = dict()
        main_results, aux_results = self.forward(x)
        main_loss_inputs, aux_loss_inputs = self.get_inputs(
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

        return loss

    def simple_test(self, x, img_metas, **kwargs):

        main_results, aux_results = self.forward(x)
        main_outs, _ = self.get_inputs(main_results,
                                       aux_results,
                                       img_metas=img_metas)

        results_list = self.get_bboxes(*main_outs, img_metas)
        return results_list

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
        # test stage
        if aux_results is None:
            (aux_cls_scores, aux_bbox_preds) = (None, None)
        else:
            aux_cls_scores, aux_bbox_preds, all_query_ids = self.pre_dqs(
                **aux_results, mlvl_priors=mlvl_priors, img_metas=img_metas)

        nms_all_cls_scores, nms_all_bbox_preds = self.dqs(
            all_cls_scores, all_bbox_preds)

        return (nms_all_cls_scores, nms_all_bbox_preds), (aux_cls_scores,
                                                          aux_bbox_preds)

    def dqs(self, all_mlvl_scores, all_mlvl_bboxes):
        ddq_bboxes = []
        ddq_scores = []
        for mlvl_bboxes, mlvl_scores in zip(all_mlvl_bboxes, all_mlvl_scores):
            if mlvl_bboxes.numel() == 0:
                return mlvl_bboxes, mlvl_scores

            det_bboxes, ddq_idxs = batched_nms(mlvl_bboxes,
                                               mlvl_scores.max(-1).values,
                                               torch.ones(len(mlvl_scores)),
                                               self.dqs_cfg)

            ddq_bboxes.append(mlvl_bboxes[ddq_idxs])
            ddq_scores.append(mlvl_scores[ddq_idxs])
        return ddq_scores, ddq_bboxes

    def pre_dqs(self,
                cls_scores_list=None,
                bbox_preds_list=None,
                mlvl_priors=None,
                img_metas=None,
                **kwargs):

        num_imgs = cls_scores_list[0].size(0)
        all_cls_scores = []
        all_bbox_preds = []
        all_query_ids = []
        for img_id in range(num_imgs):

            single_cls_score_list = select_single_mlvl(cls_scores_list,
                                                       img_id,
                                                       detach=False)
            sinlge_bbox_pred_list = select_single_mlvl(bbox_preds_list,
                                                       img_id,
                                                       detach=False)
            cls_score, bbox_pred, query_inds = self._get_topk(
                single_cls_score_list, sinlge_bbox_pred_list, mlvl_priors,
                img_metas[img_id])
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            all_query_ids.append(query_inds)
        return all_cls_scores, all_bbox_preds, all_query_ids

    def _get_topk(self, cls_score_list, bbox_pred_list, mlvl_priors, img_meta,
                  **kwargs):
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_query_inds = []
        start_inds = 0
        for level_idx, (cls_score, bbox_pred, priors, stride) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                     mlvl_priors, \
                        self.prior_generator.strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)

            binary_cls_score = cls_score.max(-1).values.reshape(-1, 1)
            if self.dqs_cfg:
                nms_pre = self.dqs_cfg.pop('nms_pre', 1000)
            else:
                if self.training:
                    nms_pre = len(binary_cls_score)
                else:
                    nms_pre = 1000
            results = filter_scores_and_topk(
                binary_cls_score, 0, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, cls_score=cls_score))
            scores, labels, keep_idxs, filtered_results = results
            keep_idxs = keep_idxs + start_inds
            start_inds = start_inds + len(cls_score)
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            cls_score = filtered_results['cls_score']
            bbox_pred = bbox_pred * stride[0]
            bbox_pred = distance2bbox(priors, bbox_pred)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)
            mlvl_query_inds.append(keep_idxs)

        return torch.cat(mlvl_scores), torch.cat(mlvl_bboxes), torch.cat(
            mlvl_query_inds)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas=None, **kwargs):

        result_list = []
        for sinlge_score, single_bbox_pred, img_meta in zip(
                cls_scores, bbox_preds, img_metas):
            img_shape = img_meta['img_shape']
            single_bbox_pred[:, 0::2].clamp_(min=0, max=img_shape[1])
            single_bbox_pred[:, 1::2].clamp_(min=0, max=img_shape[0])
            single_bbox_pred = single_bbox_pred / single_bbox_pred.new_tensor(
                img_meta['scale_factor'])
            sinlge_score = sinlge_score.flatten(0, 1)
            num_distinct_queries = min(self.num_distinct_queries,
                                       len(sinlge_score))
            scores_per_img, topk_indices = sinlge_score.topk(
                num_distinct_queries, sorted=True)
            labels_per_img = topk_indices % self.num_classes
            bboxes = single_bbox_pred[topk_indices // self.num_classes]
            bboxes = torch.cat([bboxes, scores_per_img[:, None]], dim=1)

            result_list.append((bboxes[:num_distinct_queries],
                                labels_per_img[:num_distinct_queries]))
        return result_list
