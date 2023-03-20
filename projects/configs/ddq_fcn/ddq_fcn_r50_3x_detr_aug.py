_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(type='SingleStageDetector',
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
             neck=[
                 dict(type='FPN',
                      in_channels=[256, 512, 1024, 2048],
                      out_channels=256,
                      start_level=1,
                      add_extra_convs='on_input',
                      num_outs=5),
             ],
             bbox_head=dict(type='DDQFCNHead',
                            dqs_cfg=dict(
                                type='nms',
                                iou_threshold=0.7,
                                nms_pre=1000,
                            ),
                            strides=(8, 16, 32, 64, 128),
                            num_classes=80,
                            in_channels=256,
                            norm_cfg=dict(type='GN',
                                          num_groups=32,
                                          requires_grad=True)))

# optimizer
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4 * 0.5, weight_decay=0.1)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=10, norm_type=2))
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
lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
