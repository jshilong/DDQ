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
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=10, norm_type=2))

# remove 'NumClassCheckHook'
custom_hooks = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

# augmentation strategy originates from DETR.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[[
             dict(type='Resize',
                  img_scale=[
                      (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                      (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                      (736, 1333), (768, 1333), (800, 1333)
                  ],
                  multiscale_mode='value',
                  keep_ratio=True)
         ],
                   [
                       dict(type='Resize',
                            img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True),
                       dict(type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                       dict(type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                       (576, 1333), (608, 1333), (640, 1333),
                                       (672, 1333), (704, 1333), (736, 1333),
                                       (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True)
                   ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(train=dict(pipeline=train_pipeline))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=2000,
                 warmup_ratio=0.0001,
                 step=[27, 33])

runner = dict(type='EpochBasedRunner', max_epochs=36)
