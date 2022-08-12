# dataset settings
dataset_type = 'AnomalyDataset'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)

classes = ('normal','anomaly')


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=128, efficientnet_style=True),
    #dict(type='RandomFlip', flip_prob=0.5),
    #dict(type='Resize', size=(128,128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=128, efficientnet_style=True),
    #dict(type='RandomFlip', flip_prob=0.5),
    #dict(type='Resize', size=(128,128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32, #batch size of a single gpu 
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/AnomalyDatasetTest/training',
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/AnomalyDatasetTest/valid',
        classes = classes,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/AnomalyDatasetTest/valid',
        classes = classes,
        pipeline=test_pipeline,
        test_mode=True))