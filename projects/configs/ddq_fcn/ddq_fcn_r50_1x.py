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
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4 * 0.5, weight_decay=0.1)

# remove 'NumClassCheckHook'
custom_hooks = None
