#inherit
# from mmcls.core import evaluation  # default evalution

_base_ = [
    './anomaly_custom_resnet.py', #models
    './anomaly_dataset_custom_config.py', #dataset
    '../_base_/default_runtime.py' #runtime
    #schedules = optimizer
]

# update runtime ===========================
# log config 
log_level = 'INFO' # CRITICAL , ERROR , WARNING , INFO , DEBUG , NOTEST 

log_config = dict(
    interval=50, #for test, 100 또는 50으로 낮추기 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook') # dict(type='TensorboardLoggerHook') 
    ])

# Added ===========================
# Set evaluation 
evaluation = dict(
    interval=1,
    metric='accuracy', # metric=['bbox', 'segm']
    metric_options={'topk': (1, )}
)

# Set optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# Set customized learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])

runner = dict(type='EpochBasedRunner', max_epochs=10) #10 

