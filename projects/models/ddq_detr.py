import copy
from typing import Dict, List, Tuple

import torch
from mmcv.ops import MultiScaleDeformableAttention, batched_nms
from mmdet.models import (DINO, MLP, DeformableDETR,
                          DeformableDetrTransformerDecoder, DINOHead,
                          coordinate_to_encoding, inverse_sigmoid)
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor, nn
from torch.nn.init import normal_

from .utils import AuxLossModule, align_tensor


class DDQTransformerDecoder(DeformableDetrTransformerDecoder):

    def _init_layers(self) -> None:
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def be_distinct(self, ref_points, query, self_attn_mask, lid):
        num_imgs = len(ref_points)
        dis_start, num_dis = self.cache_dict['dis_query_info']
        # shape of self_attn_mask
        # (batch⋅num_heads, num_queries, embed_dims)
        dis_mask = self_attn_mask[:,dis_start: dis_start + num_dis, \
                   dis_start: dis_start + num_dis]
        # cls_branches from DDQDETRHead
        scores = self.cache_dict['cls_branches'][lid](
            query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
        proposals = ref_points[:, dis_start:dis_start + num_dis]
        proposals = bbox_cxcywh_to_xyxy(proposals)

        attn_mask_list = []
        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            attn_mask = ~dis_mask[img_id * self.cache_dict['num_heads']][0]
            # distinct query inds in this layer
            ori_index = attn_mask.nonzero().view(-1)
            _, keep_idxs = batched_nms(single_proposals[ori_index],
                                       single_scores[ori_index],
                                       torch.ones(len(ori_index)),
                                       self.cache_dict['dqs_cfg'])

            real_keep_index = ori_index[keep_idxs]

            attn_mask = torch.ones_like(dis_mask[0]).bool()
            # such a attn_mask give best result
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False

            attn_mask = attn_mask[None].repeat(self.cache_dict['num_heads'], 1,
                                               1)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.cat(attn_mask_list)
        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, dis_start: dis_start + num_dis, \
            dis_start: dis_start + num_dis] = attn_mask
        # will be used in loss and inference
        self.cache_dict['distinct_query_mask'].append(~attn_mask)
        return self_attn_mask

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:

        intermediate = []
        intermediate_reference_points = [reference_points]
        self.cache_dict['distinct_query_mask'] = []
        if self_attn_mask is None:
            self_attn_mask = torch.zeros(
                (query.size(1), query.size(1))).bool().cuda()
        # shape is (batch*number_heads, num_queries, num_queries)
        self_attn_mask = self_attn_mask[None].repeat(
            len(query) * self.cache_dict['num_heads'], 1, 1)
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(query,
                          query_pos=query_pos,
                          value=value,
                          key_padding_mask=key_padding_mask,
                          self_attn_mask=self_attn_mask,
                          spatial_shapes=spatial_shapes,
                          level_start_index=level_start_index,
                          valid_ratios=valid_ratios,
                          reference_points=reference_points_input,
                          **kwargs)

            if not self.training:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if lid < (len(self.layers) - 1):
                    self_attn_mask = self.be_distinct(reference_points, query,
                                                      self_attn_mask, lid)

            else:
                num_dense = self.cache_dict['num_dense_queries']
                tmp_dense = reg_branches[lid](query[:, :-num_dense])
                tmp = self.aux_reg_branches[lid](query[:, -num_dense:])
                tmp = torch.cat([tmp_dense, tmp], dim=1)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if lid < (len(self.layers) - 1):
                    self_attn_mask = self.be_distinct(reference_points, query,
                                                      self_attn_mask, lid)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class DDQDETR(DINO):

    def __init__(self,
                 *args,
                 dqs_cfg=dict(type='nms', iou_threshold=0.8),
                 **kwargs):
        self.decoder_cfg = kwargs['decoder']
        self.dqs_cfg = dqs_cfg
        super().__init__(*args, **kwargs)

        # a share dict in all moduls
        # pass some intermediate results and config parameters
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        # first element is the start index of matching queries
        # second element is the number of matching queries
        self.cache_dict['dis_query_info'] = [0, 0]

        # mask for distinct queries in each decoder layer
        self.cache_dict['distinct_query_mask'] = []
        # pass to decoder do the dqs
        self.cache_dict['cls_branches'] = self.bbox_head.cls_branches
        # Used to construct the attention mask after dqs
        self.cache_dict['num_heads'] = self.encoder.layers[
            0].self_attn.num_heads
        # pass to decoder to do the dqs
        self.cache_dict['dqs_cfg'] = self.dqs_cfg

    def init_weights(self) -> None:
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        normal_(self.level_embed)

    def _init_layers(self) -> None:
        super(DDQDETR, self)._init_layers()
        self.decoder = DDQTransformerDecoder(**self.decoder_cfg)
        self.query_embedding = None
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        if self.training:
            # -1 is the aux head for the encoder
            dense_enc_outputs_class = self.bbox_head.cls_branches[-1](
                output_memory)
            dense_enc_outputs_coord_unact = self.bbox_head.reg_branches[-1](
                output_memory) + output_proposals

        topk = self.num_queries
        dense_topk = int(topk * 1.5)

        proposals = enc_outputs_coord_unact.sigmoid()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        scores = enc_outputs_class.max(-1)[0].sigmoid()

        if self.training:
            dense_proposals = dense_enc_outputs_coord_unact.sigmoid()
            dense_proposals = bbox_cxcywh_to_xyxy(dense_proposals)
            dense_scores = dense_enc_outputs_class.max(-1)[0].sigmoid()

        num_imgs = len(scores)
        topk_score = []
        topk_coords_unact = []
        query = []

        dense_topk_score = []
        dense_topk_coords_unact = []
        dense_query = []

        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            _, keep_idxs = batched_nms(single_proposals, single_scores,
                                       torch.ones(len(single_scores)),
                                       self.cache_dict['dqs_cfg'])
            if self.training:
                dense_single_proposals = dense_proposals[img_id]
                dense_single_scores = dense_scores[img_id]
                # sort according the score
                _, dense_keep_idxs = batched_nms(
                    dense_single_proposals, dense_single_scores,
                    torch.ones(len(dense_single_scores)), None)

                dense_topk_score.append(dense_enc_outputs_class[img_id]
                                        [dense_keep_idxs][:dense_topk])
                dense_topk_coords_unact.append(
                    dense_enc_outputs_coord_unact[img_id][dense_keep_idxs]
                    [:dense_topk])

            topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])
            topk_coords_unact.append(
                enc_outputs_coord_unact[img_id][keep_idxs][:topk])

            map_memory = self.query_map(memory[img_id].detach())
            query.append(map_memory[keep_idxs][:topk])
            if self.training:
                dense_query.append(map_memory[dense_keep_idxs][:dense_topk])

        topk_score = align_tensor(topk_score, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        query = align_tensor(query, topk)
        if self.training:
            dense_topk_score = align_tensor(dense_topk_score)
            dense_topk_coords_unact = align_tensor(dense_topk_coords_unact)

            dense_query = align_tensor(dense_query)
            num_dense_queries = dense_query.size(1)
        if self.training:
            query = torch.cat([query, dense_query], dim=1)
            topk_coords_unact = torch.cat(
                [topk_coords_unact, dense_topk_coords_unact], dim=1)

        topk_anchor = topk_coords_unact.sigmoid()
        if self.training:
            dense_topk_anchor = topk_anchor[:, -num_dense_queries:]
            topk_anchor = topk_anchor[:, :-num_dense_queries]

        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
            ori_size = dn_mask.size(-1)
            new_size = dn_mask.size(-1) + num_dense_queries

            new_dn_mask = dn_mask.new_ones((new_size, new_size)).bool()
            dense_mask = torch.zeros(num_dense_queries,
                                     num_dense_queries).bool()
            self.cache_dict['dis_query_info'] = [dn_label_query.size(1), topk]

            new_dn_mask[ori_size:, ori_size:] = dense_mask
            new_dn_mask[:ori_size, :ori_size] = dn_mask
            dn_meta['num_dense_queries'] = num_dense_queries
            dn_mask = new_dn_mask
            self.cache_dict['num_dense_queries'] = num_dense_queries
            self.decoder.aux_reg_branches = self.bbox_head.aux_reg_branches

        else:
            self.cache_dict['dis_query_info'] = [0, topk]
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(query=query,
                                   memory=memory,
                                   reference_points=reference_points,
                                   dn_mask=dn_mask)
        head_inputs_dict = dict(enc_outputs_class=topk_score,
                                enc_outputs_coord=topk_anchor,
                                dn_meta=dn_meta) if self.training else dict()
        if self.training:
            head_inputs_dict['aux_enc_outputs_class'] = dense_topk_score
            head_inputs_dict['aux_enc_outputs_coord'] = dense_topk_anchor

        return decoder_inputs_dict, head_inputs_dict


@MODELS.register_module()
class DDQDETRHead(DINOHead):

    def __init__(self, *args, dn_loss=True, aux_num_pos=4, **kwargs):
        self.dn_loss = dn_loss
        super(DDQDETRHead, self).__init__(*args, **kwargs)
        self.aux_loss_for_dense = AuxLossModule(train_cfg=dict(assigner=dict(
            type='TopkHungarianAssigner', topk=aux_num_pos),
                                                               alpha=1,
                                                               beta=6), )

    def _init_layers(self) -> None:
        super(DDQDETRHead, self)._init_layers()
        # aux head for dense queries on encoder feature map
        self.cls_branches.append(copy.deepcopy(self.cls_branches[-1]))
        self.reg_branches.append(copy.deepcopy(self.reg_branches[-1]))

        # self.num_pred_layer is 7
        # aux head for dense queries in decoder
        self.aux_cls_branches = nn.ModuleList([
            copy.deepcopy(self.cls_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])
        self.aux_reg_branches = nn.ModuleList([
            copy.deepcopy(self.reg_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""

        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m.bias, bias_init)
        for m in self.aux_cls_branches:
            nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)

        for m in self.aux_reg_branches:
            constant_init(m[-1], 0, bias=0)

        for m in self.aux_reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        if self.training:
            num_dense = self.cache_dict['num_dense_queries']
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            if self.training:
                dense_hidden_state = hidden_state[:, -num_dense:]
                hidden_state = hidden_state[:, :-num_dense]

            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if self.training:
                dense_outputs_class = self.aux_cls_branches[layer_id](
                    dense_hidden_state)
                dense_tmp_reg_preds = self.aux_reg_branches[layer_id](
                    dense_hidden_state)
                outputs_class = torch.cat([outputs_class, dense_outputs_class],
                                          dim=1)
                tmp_reg_preds = torch.cat([tmp_reg_preds, dense_tmp_reg_preds],
                                          dim=1)

            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             aux_enc_outputs_class=None,
             aux_enc_outputs_coord=None) -> dict:

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        aux_enc_outputs_coord = bbox_cxcywh_to_xyxy(aux_enc_outputs_coord)
        aux_enc_outputs_coord_list = []
        for img_id in range(len(aux_enc_outputs_coord)):
            det_bboxes = aux_enc_outputs_coord[img_id]
            img_shape = batch_img_metas[img_id]['img_shape']
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            aux_enc_outputs_coord_list.append(det_bboxes)
        aux_enc_outputs_coord = torch.stack(aux_enc_outputs_coord_list)
        aux_loss = self.aux_loss_for_dense.loss(
            aux_enc_outputs_class.sigmoid(), aux_enc_outputs_coord,
            [item.bboxes for item in batch_gt_instances],
            [item.labels for item in batch_gt_instances], batch_img_metas)
        for k, v in aux_loss.items():
            losses[f'aux_enc_{k}'] = v

        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        num_dense_queries = dn_meta['num_dense_queries']
        num_layer = all_layers_matching_bbox_preds.size(0)
        dense_all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :,
                                                                              -num_dense_queries:]
        dense_all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :,
                                                                              -num_dense_queries:]

        all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :, :
                                                                        -num_dense_queries]
        all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :, :
                                                                        -num_dense_queries]

        loss_dict = self.loss_for_distinct_queries(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        if enc_cls_scores is not None:

            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i

        for l_id in range(num_layer):
            cls_scores = dense_all_layers_matching_cls_scores[l_id].sigmoid()
            bbox_preds = dense_all_layers_matching_bbox_preds[l_id]

            bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds)
            bbox_preds_list = []
            for img_id in range(len(bbox_preds)):
                det_bboxes = bbox_preds[img_id]
                img_shape = batch_img_metas[img_id]['img_shape']
                det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
                det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
                bbox_preds_list.append(det_bboxes)
            bbox_preds = torch.stack(bbox_preds_list)
            aux_loss = self.aux_loss_for_dense.loss(
                cls_scores, bbox_preds,
                [item.bboxes for item in batch_gt_instances],
                [item.labels for item in batch_gt_instances], batch_img_metas)
            for k, v in aux_loss.items():
                loss_dict[f'{l_id}_aux_{k}'] = v

        return loss_dict

    def loss_for_distinct_queries(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self._loss_for_distinct_queries_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            [i for i in range(len(all_layers_bbox_preds))],
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def _loss_for_distinct_queries_single(self, cls_scores, bbox_preds, l_id,
                                          batch_gt_instances, batch_img_metas):

        num_imgs = cls_scores.size(0)
        if 0 < l_id:
            batch_mask = [
                self.cache_dict['distinct_query_mask'][l_id - 1][
                    img_id * self.cache_dict['num_heads']][0]
                for img_id in range(num_imgs)
            ]
        else:
            batch_mask = [
                torch.ones(len(cls_scores[i])).bool().cuda()
                for i in range(num_imgs)
            ]
        # only select the distinct queries in decoder for loss
        cls_scores_list = [
            cls_scores[i][batch_mask[i]] for i in range(num_imgs)
        ]
        bbox_preds_list = [
            bbox_preds[i][batch_mask[i]] for i in range(num_imgs)
        ]
        cls_scores = torch.cat(cls_scores_list)

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds_list):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = torch.cat(bbox_preds_list)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def predict_by_feat(self,
                        layer_cls_scores: Tensor,
                        layer_bbox_preds: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:

        cls_scores = layer_cls_scores[-1]
        bbox_preds = layer_bbox_preds[-1]

        num_imgs = cls_scores.size(0)
        # -1 is last layer input query mask

        batch_mask = [
            self.cache_dict['distinct_query_mask'][-1][
                img_id * self.cache_dict['num_heads']][0]
            for img_id in range(num_imgs)
        ]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id][batch_mask[img_id]]
            bbox_pred = bbox_preds[img_id][batch_mask[img_id]]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
