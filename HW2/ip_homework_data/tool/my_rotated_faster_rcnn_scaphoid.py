_base_ = [
    '../_base_/datasets/dotav1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'  # DOTA 常用設定

model = dict(
    type='RotatedFasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0
        )
    ),
    roi_head=dict(
        type='RotatedStandardRoIHead',
        version=angle_version,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,   # <--- 只有一類: Scaphoid
            bbox_coder=dict(
                type='DeltaXYWHAHBBoxCoder',
                angle_range=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=(0., 0., 0., 0., 0.),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)
            ),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0
            )
        )
    ),
    # 訓練配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    # 測試配置
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000
        )
    )
)

# ------------------- Dataset / Pipeline --------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 與預訓練模型相符
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # 會讀 DOTA 標籤
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],  # 訓練階段的翻轉比例
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    # 不需要 Lambda 步驟，因為 DefaultFormatBundle 已處理
]

test_pipeline = [
    dict(
        type='LoadImageFromFile'
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,  # 測試階段通常不進行翻轉
        transforms=[
            dict(type='RResize'),
            dict(type='RRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'pad_shape', 'scale_factor', 'flip',
                           'flip_direction')
            )
        ]
    )
]

data = dict(
    samples_per_gpu=2,  # 訓練階段使用的批次大小
    workers_per_gpu=2,
    train=dict(
        type='DOTADataset',
        ann_file='../scaphoid_detection/labelTxt',  # 您的 labelTxt 路徑
        img_prefix='../scaphoid_detection/images',  # 您的影像路徑
        pipeline=train_pipeline,
        version=angle_version,
        classes=('Scaphoid', )   # <--- 單一類：Scaphoid
    ),
    val=dict(
        type='DOTADataset',
        ann_file='../scaphoid_detection/labelTxt',  # 可以分開訓練和驗證集
        img_prefix='../scaphoid_detection/images',
        pipeline=test_pipeline,
        version=angle_version,
        classes=('Scaphoid', )
    ),
    test=dict(
        type='DOTADataset',
        ann_file='../scaphoid_detection/labelTxt',
        img_prefix='../scaphoid_detection/images',
        pipeline=test_pipeline,
        version=angle_version,
        classes=('Scaphoid', )
    )
)

# 添加單獨的 test_dataloader 配置
test_dataloader = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    shuffle=False
)

optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
)

# 其餘 (lr_config, runner, etc.) 依需求自行在 _base_/*.py 中或此處修改

# 確保工作目錄設置正確
work_dir = './work_dirs/my_rotated_faster_rcnn_scaphoid'
auto_resume = False
gpu_ids = range(0, 1)
