# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-base',
        img_size=384,
        patch_size=16,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=2,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    # Change to the path of the pretrained model
    # init_cfg=dict(type='Pretrained', checkpoint=''),
)

