# Segmentation models for NIA Project

This repository is for verifying the NIA dataset with recent models at Korea University.

1. PointRend : Image Segmentation as Rendering (ArXiv:1912.08193)
2. YOLACT : YOLACT: Real-time Instance Segmentation (ICCV2019)

The details are in each folder (PointRend, YOLACT)

## Envs

All models in this repository are tested with

python = 3.9

pyTorch = 1.11 with cuda 11.3

### Dependencies

```

Python>=3.7, PyTorch>=1.1, numpy, cffi, imageio, opencv-python, tqdm

```

### Conda (env name : seg_3)

```
conda env create --file environment.yaml
```

## Structure

```
- seg_data
    - coco
        - annotations
        - test2017
        - train2017
        - val2017
    - Pascal
        - sbd
        - img
- PointRend
    - dataset
        - coco (ln)
    - args_configs
    - model_configs
- yolact
    - data
        - coco (ln)
        - scripts
```

## Dataset

|  DataSet  |                              Description                              |
| :--------: | :--------------------------------------------------------------------: |
|  MS-COCO  |                        https://cocodataset.org/                        |
| Pascal SBD |         http://home.bharathh.info/pubs/codes/SBD/download.html         |
|            | https://drive.google.com/file/d/1ExrRSPVctHW8Nxrn0SofU1lVhK5Wn0_S/view |
| Cityscapes |                  https://www.cityscapes-dataset.com/                  |

You can download all dataset, which should be unziped in seg_data folder.
You can change the dataset path as your convinience, and the arguments for path should be modified.
For your convinience, please symbolic link set_data in each folder that you want to run.
Here are the examples.

```
cd ./PointRend/dataset
ln -s ../../seg_data/coco .

```

MS-COCO dataset formating is recommended when you are using your own dataset.
You can learn how to make it with ./PointRend/dataset/README.md.

## QuickStart

There are various setting for training and testing models, which is described the README files in each folders.
Here are the quick start manuals to train and evaluate models.

### PontRend

#### Training

To train a model with N GPUs run :

```bash
cd ./PointRend
python train_net.py --config-file ./args_configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus N
```

#### Evaluation

```bash
cd ./PointRend
python train_net.py --config-file ./args_configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

You can donwload the pretrained models, and the links are discribed in  ./PointRend/dataset/README.md.

### YOLOACT

There are various useful scripts for running YOLOACT in ./yoloact/data/scripts.
Here are the basic commands for training and evaluating YOLOACT.

#### Training

```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

#### Evaluation

Here are our YOLACT models (released on April 5th, 2019) along with their FPS on a Titan Xp and mAP on `test-dev`:

| Image Size |   Backbone   | FPS | mAP | Weights                                                                                                           |                                                                                                                                  |
| :--------: | :-----------: | :--: | :--: | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
|    550    | Resnet50-FPN | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
|    550    | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw) |
|    550    | Resnet101-FPN | 33.5 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg) |
|    700    | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)     | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/Eagg5RSc5hFEhp7sPtvLNyoBjhlf2feog7t8OQzHKKphjw) |

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

#### Custom Datasets

You can also train on your own dataset by following these steps:

- Create a COCO-style Object Detection JSON annotation file for your dataset. The specification for this can be found [here](http://cocodataset.org/#format-data). Note that we don't use some fields, so the following may be omitted:
  - `info`
  - `liscense`
  - Under `image`: `license, flickr_url, coco_url, date_captured`
  - `categories` (we use our own format for categories, see below)
- Create a definition for your dataset under `dataset_base` in `data/config.py` (see the comments in `dataset_base` for an explanation of each field):

```Python
my_custom_dataset = dataset_base.copy({
    'name': 'My Dataset',

    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',

    'has_gt': True,
    'class_names': ('my_class_id_1', 'my_class_id_2', 'my_class_id_3', ...)
})
```

- A couple things to note:
  - Class IDs in the annotation file should start at 1 and increase sequentially on the order of `class_names`. If this isn't the case for your annotation file (like in COCO), see the field `label_map` in `dataset_base`.
  - If you do not want to create a validation split, use the same image path and annotations file for validation. By default (see `python train.py --help`), `train.py` will output validation mAP for the first 5000 images in the dataset every 2 epochs.
- Finally, in `yolact_base_config` in the same file, change the value for `'dataset'` to `'my_custom_dataset'` or whatever you named the config object above. Then you can use any of the training commands in the previous section.

#### Creating a Custom Dataset from Scratch

See [this nice post by @Amit12690](https://github.com/dbolya/yolact/issues/70#issuecomment-504283008) for tips on how to annotate a custom dataset and prepare it for use with YOLACT.
