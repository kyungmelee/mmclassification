# model settings
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

model = dict(
    type='ImageClassifier',
    backbone = dict(type='MobileNetV2', widen_factor=1.0, 
    init_cfg =dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone') ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))


#Z:\Open-mmlab\mmclassification-test\docs\en\tutorials\MMClassification_python.ipynb
