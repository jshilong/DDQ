from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class DDQRCNN(TwoStageDetector):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        losses = dict()
        x = self.extract_feat(img)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        rpn_losses, imgs_whwh, distinc_query_dict = \
            self.rpn_head.forward_train(
                rpn_x,
                img_metas,
                gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                **kwargs)
        proposals = distinc_query_dict['proposals']
        object_feats = distinc_query_dict['object_feats']

        for k, v in rpn_losses.items():
            losses[f'rpn_{k}'] = v

        roi_losses = self.roi_head.forward_train(
            roi_x,
            proposals,
            object_feats,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=True):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        rpn_losses, imgs_whwh, distinc_query_dict = \
            self.rpn_head.simple_test_rpn(
                rpn_x,
                img_metas,
                )
        proposals = distinc_query_dict['proposals']
        object_feats = distinc_query_dict['object_feats']

        results = self.roi_head.simple_test(roi_x,
                                            proposals,
                                            object_feats,
                                            img_metas,
                                            imgs_whwh=imgs_whwh,
                                            rescale=rescale)
        return results
