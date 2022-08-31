import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = '../configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cpu')

import os

list_train_images = os.listdir('./200224/training/images')

import json 

with open('./200224/training/annotations/instances_default.json','r') as jsonfile:
    annot_data = json.load(jsonfile)

@DATASETS.register_module()
class HatchingDataset(CustomDataset):

    CLASSES = ('Juvenile',)

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
        # image_list = list_train_images        
        data_infos = []
        #print(list_train_images)
        # convert annotations to middle format
        for fname in image_list:
            bboxes = []
            bbox_names=[]
            filepath = './200224/training/images/{0}'.format(fname)
            print(filepath)
            image = mmcv.imread(filepath)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{fname}', width=width, height=height)
    
            # load annotations
             # load annotations
            image_id = [i['id'] for i in annot_data['images'] if i['file_name']==fname][0]
            bounding_boxes = [b['bbox'] for b in annot_data['annotations'] if b['image_id']==image_id]
            
            for bbox_coco in bounding_boxes:
                bbox_kitty = [bbox_coco[0], bbox_coco[1], bbox_coco[0]+bbox_coco[2], bbox_coco[1]+bbox_coco[3]]                                                      
                bboxes.append(bbox_kitty)            
                bbox_names.append('Juvenile')

            # label_prefix = self.img_prefix.replace('image_2', 'label_2')
            # lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
    
            # content = [line.strip().split(' ') for line in lines]

            # bbox_names = [x[0] for x in content]
            # bboxes = [[float(info) for info in x[4:8]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


# read config
from mmcv import Config
cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')


# adjust config

from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'HatchingDataset'
cfg.data_root = '200224/'

cfg.data.test.type = 'HatchingDataset'
cfg.data.test.data_root = '200224/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'training/images'

cfg.data.train.type = 'HatchingDataset'
cfg.data.train.data_root = '200224/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'training/images'

cfg.data.val.type = 'HatchingDataset'
cfg.data.val.data_root = '200224/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'training/images'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 1

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.runner.max_epochs = 4
# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


