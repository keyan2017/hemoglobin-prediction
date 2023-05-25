import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import numpy as np

# 导入模型参数
cfg = mmcv.Config.fromfile('configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_jn_test.py')
cfg.model.pretrained = None

# 构建化模型和加载检查点卡
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'work_dirs/epoch_24.pth')

# 测试单张图片
img = mmcv.imread('/home/xulinquan/data/3.jpg')
result = inference_detector(model, img, cfg)
#print(result)
#show_result(img, result)
if isinstance(result, tuple):
    bbox_result, segm_result = result
else:
    bbox_result, segm_result = result, None
bboxes = np.vstack(bbox_result)
'''
# draw segmentation masks
if segm_result is not None:
    segms = mmcv.concat_list(segm_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    for i in inds:
        color_mask = np.random.randint(
            0, 256, (1, 3), dtype=np.uint8)
        mask = maskUtils.decode(segms[i]).astype(np.bool)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5
'''
# draw bounding boxes
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
]
labels = np.concatenate(labels)
print(bboxes)

# 测试（多张）图片列表
'''
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
'''
