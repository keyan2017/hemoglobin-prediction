#from mmdet.apis import init_detector, inference_detector, show_result
import os
import shutil
import argparse
import sys
import torch
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--config', dest='config',help='config_file',default='/home/xulinquan/mmdet/configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_jn_test.py', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/home/xulinquan/mmdet/work_dirs/epoch_24.pth', type=str)
    '''
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    '''
    args = parser.parse_args()
    return args
    

def main():
    #args = parse_args()
    config_file = '/home/xulinquan/mmdet/configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_jn_test.py'
    checkpoint_file = '/home/xulinquan/mmdet/work_dirs/epoch_24.pth'

    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.pretrained = None

    # constract model
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model,checkpoint_file )

    # test single img
    bp = '/home/xulinquan/data/b'
    nf='/home/xulinquan/data/a'
    for a_file in os.listdir(bp):
        i_file = bp + '/'+a_file
        img = mmcv.imread(i_file)
        result = inference_detector(model, img, cfg)
        #print(result)
        #show_result(img, result)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        for b in bboxes:
            if b[4]>0.7:
                shutil.copyfile(i_file,nf+'/'+a_file)
                print(b[4])
        print(bboxes)


if __name__ == '__main__':
    main()