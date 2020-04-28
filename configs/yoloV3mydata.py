input_size = 416
model = dict(
    type='SingleStageDetector',
    #pretrained='./weights/darknet53_weights.pth',
    pretrained=None,
    backbone=dict(
        type='DarkNet',
        input_size=input_size,
        depth=53,
        frozen_stages=3,
        out_indices=(2, 3, 4),
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='YoloHead',
        in_channels=[256, 512, 1024],
        num_classes=3,
        input_size=input_size,
        anchors=[[10,13],[16,18],[33,23],[30,45],[62,45],[59,119],[116,90],[156,198],[373,326]],
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)

dataset_type = 'myVOCDataset'
data_root = 'data/AI_data/VOCdevkit/'
img_norm_cfg = dict(mean=[127], std=[1], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),

    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 4)),
    dict(
        type='Photo2Gray'
    ),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Photo2Gray'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2020/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2020/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2020/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2020/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2020/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2020/',
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', step=[80, 90])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
total_epochs = 100
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
work_dir = './work_dirs/yolov3my'
load_from = './work_dirs/yolov3/latest.pth'
resume_from = None
workflow = [('train', 1)]
checkpoint = work_dir + '/latest.pth'
