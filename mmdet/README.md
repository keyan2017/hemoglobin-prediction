
# mmdetection

## Introduction
The code implementation for our paper is based on [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).



## License

This project is released under the [Apache 2.0 license](LICENSE).



## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Train
```
single-gpu testing
python tools/train.py configs/cascade_mask_rcnn_x101_64x4d_fpn_1x_jn_train.py
```
## Test
```
single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```



