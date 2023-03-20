_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

num_stages = 2
model = dict(
    type='DDQRCNN',
    backbone=dict(type='ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=True),
                  norm_eval=True,
                  style='pytorch',
                  init_cfg=dict(type='Pretrained',
                                checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        # top 5 level for rpn
        # p2 only for rcnn
        num_outs=6),
    rpn_head=dict(type='DDQFCNRPN',
                  num_distinct_queries=300,
                  num_classes=80,
                  in_channels=256,
                  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    roi_head=dict(type='SparseRoIHead',
                  num_stages=num_stages,
                  stage_loss_weights=[1] * num_stages,
                  proposal_feature_channel=256,
                  bbox_roi_extractor=dict(
                      type='SingleRoIExtractor',
                      roi_layer=dict(type='RoIAlign',
                                     output_size=7,
                                     sampling_ratio=2),
                      out_channels=256,
                      featmap_strides=[4, 8, 16, 32, 64, 128]),
                  bbox_head=[
                      dict(type='DIIHead',
                           num_classes=80,
                           num_ffn_fcs=2,
                           num_heads=8,
                           num_cls_fcs=1,
                           num_reg_fcs=3,
                           feedforward_channels=2048,
                           in_channels=256,
                           dropout=0.0,
                           ffn_act_cfg=dict(type='ReLU', inplace=True),
                           dynamic_conv_cfg=dict(type='DynamicConv',
                                                 in_channels=256,
                                                 feat_channels=64,
                                                 out_channels=256,
                                                 input_feat_shape=7,
                                                 act_cfg=dict(type='ReLU',
                                                              inplace=True),
                                                 norm_cfg=dict(type='LN')),
                           loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                           loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                           loss_cls=dict(type='FocalLoss',
                                         use_sigmoid=True,
                                         gamma=2.0,
                                         alpha=0.25,
                                         loss_weight=2.0),
                           bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                           clip_border=False,
                                           target_means=[0., 0., 0., 0.],
                                           target_stds=[0.5, 0.5, 1., 1.]))
                      for _ in range(num_stages)
                  ]),
    # training and testing settings
    train_cfg=dict(rpn=None,
                   rcnn=[
                       dict(assigner=dict(type='HungarianAssigner',
                                          cls_cost=dict(type='FocalLossCost',
                                                        weight=2.0),
                                          reg_cost=dict(type='BBoxL1Cost',
                                                        weight=5.0),
                                          iou_cost=dict(type='IoUCost',
                                                        iou_mode='giou',
                                                        weight=2.0)),
                            sampler=dict(type='PseudoSampler'),
                            pos_weight=1) for _ in range(num_stages)
                   ]),
    test_cfg=dict(rpn=dict(), rcnn=dict(max_per_img=300)))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.05)

# remove 'NumClassCheckHook'
custom_hooks = None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
)
