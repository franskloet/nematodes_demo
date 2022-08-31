_base_ = '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3)),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# learning policy
optimizer = dict(lr = 0.02/8)
lr_config = dict(warmup = None)
log_config = dict(interval=10)         

evaluation = dict(metric = ['bbox'],
                interval = 10)
checkpoint_config = dict(interval = 10)
# classes = ('Juvenile',)
classes = ('Hatched', 'Unhatched', 'Juvenile')

data_root = "./mm_images"
data = dict(
    workers_per_gpu=3,
    samples_per_gpu=3,
    train=dict(
        pipeline=train_pipeline,
        img_prefix=f'{data_root}/training/images',
        classes=classes,
        ann_file=f'{data_root}/training/annotations/instances_default_juvenile_unhatched.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix=f'{data_root}/training/images',
        classes=classes,
        ann_file=f'{data_root}/training/annotations/instances_default_juvenile_unhatched.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix=f'{data_root}/training/images',
        classes=classes,
        ann_file=f'{data_root}/training/annotations/instances_default_juvenile_unhatched.json'))


seed = 0
runner = dict(max_epochs=500)
# data_root = '../data/2022_02_14_coco_format_select_vids/'
load_from = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
work_dir = './work_dir/juvenile_hatched_unhatched_b_3'
