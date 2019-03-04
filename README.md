# Semantic Segmentation (Keras) 
___
**Keras implementation of semantic segmentation FCNs**
**Update Logs**
> **2019-02-22**:  implemented several common FCNs and support Geo-tiff Images (especially for remote sensing images).
> 

**TODO**
> - [ ] More SOTA FCN architectures.
> - [ ] Support different output strides.
> - [ ] Support different open data sets like VOC, CityScapes, ADE20K, MSCOCO, etc.
> - [ ] More flexible in data format.


**Backbones(Encoders) that have been implemented**
> - [x] ResNet_v2 (including ResNet_v2-50, ResNet_v2-101, ResNet_v2-150, ResNet_v2-200)
> - [x] ResNet_v2_separable (including ResNet_v2-50, ResNet_v2-101, ResNet_v2-150, ResNet_v2-200)
> - [x] VGG (inclufing VGG-16, VGG-19)
> - [x] Xception-41

**Backbones to be implemented**
> - [ ] DenseNet


**FCNs that have been implemented**
> - [x] [FCN][FCN_paper] (including FCN-8s, FCN-16s, FCN-32s)
> - [x] [SegNet][SegNet_paper]
> - [x] U-Net, Res U-Net, Mobile U-Net
> - [x] [PSPNet][PSPNet_paper]
> - [x] [RefineNet][RefineNet_paper]
> - [x] [Deeplab v3][Deeplab_v3_paper]
> - [x] [Deeplab v3+][Deeplab_v3p_paper]
> - [x] [Dense ASPP][DenseASPP_paper]

**FCNs to be implemented**
> - [ ] ICNet

---
### Folder Structures
.
├── core
│        ├── __init__.py
│        ├── configures.py
│        ├── encoder
│        |        ├── __init__.py
│        |        ├── resnet_v2.py
│   　   |        ├── resnet_v2_separable.py
│        |        ├── vggs.py
│        |        └── xceptions.py
│        |
│        ├── nets
│        |        ├── __init__.py
│        |        ├── fcns.py
│        |        ├── segnets.py
│        |        ├── unets.py
│        |        ├── pspnets.py
│        |        ├── refinenets.py
│        |        ├── deeplabs.py
│        |        └── dense_aspp.py
│        |
│        └── utils
│        ├── __init__.py
│        |
│        ├── data_utils
│        |        ├── __init__.py
│        |        ├── image_io_utils.py
│        |        ├── label_transform_utils.py
│        |        ├── image_augmentation_utils.py
│        |        └── generate_dataset_utils.py
│        |
│        ├── loss_utils.py
│        ├── metric_utils.py
│        ├── net_utils.py
│        ├── model_utils.py
│        ├── training_utils.py
│        ├── predicting_utils.py
│        └── visualize_utils.py
│   
├── data
│   
├── examples
│        ├── __init__.py
│        ├── s1_generate_datasets.py
│        ├── s2_training.py
│        └── s3_predicting.py
│   
├── LICENSE
|
└── README.md

---
### Running environment
The source code was compiled in a Windows 10 platform using Python 3.6.
The dependencies include:
> tensorflow-gpu: 1.9, backend
> Keras: 2.2.4, framework
> opencv: 4.0, for image I/O
> PIL: used for image I/O
> numpy: used for array operations
> matplotlib: used to visualize images
> tqdm: used to log iterations
> GDAL: used for geo-spatial image I/O
> scikit-learn: used for metric evaluation

---
### Usage
#### 1. Generate data sets
This section introduces how to generate data sets for model training and validation. 
#### 2. Training models

The structure of the training configuration file (*.json formatted) was illustrated below. We can index a set of training args through a `model name` and a `data set name`.
.
├── model name 1
|   ├── data set name 1
|   |   └── training args
|   |
|   ├── data set name 2
|   |   └── training args
|   |
|   └── data set name n
|       └── training args
|
├── model name 2
|   ├── ...
A demo of training configuration file can be found in the following. Here, parameters for a `deeplab_v3p` model on the `ade20k` data set was set.
> `model_name`: name of the FCN model, { 'FCN-8s', 'FCN-16s', 'FCN-32s', 'SegNet', 'UNet', 'ResUNet', 'MobileUNet', 'PSPNet', 'RefineNet', 'Deeplab_v3', 'Deeplab_v3p', 'DenseASPP'}. 

```json
{
  "deeplab_v3p": {
    "ade20k": {
      "model_name": "deeplab_v3p",
      "old_model_version": "deeplab_v3p_ade20k",
      "new_model_version": "deeplab_v3p_ade20k",
      "training_samples_dir": "E:/SegData/ade20k/data_ori/train",
      "validation_samples_dir": "E:/SegData/ade20k/data_ori/val",
      "workspace": "E:/SegData/ade20k",
      "label_is_gray": 1,
      "ingore_label": 0,
      "image_width": 256,
      "image_height": 256,
      "image_channel": 3,
      "n_class": 150,
      "colour_mapping_path": "E:/SemanticSegmentation_Keras/configures/colour_mapping.json",
      "init_learning_rate": 0.001,
      "optimizer": "Adam",
      "verbose": 1,
      "batch_size": 2,
      "epoch": 50,
      "steps_per_epoch": 400,
      "steps_per_epoch_val": 200,
      "init_filters": 64,
      "dropout": 0.4,
      "loss": "crossentropy",
      "metric_name": "acc",
      "encoder_name": "xception_41",
      "callbacks": {
        "early_stop": {
          "patience": 25
        }
      }
    }
  }
}
```



#### 3. Applying Predicting 
#### 4. Evaluation 

---
### Contact
Penghua Liu (liuph3@mail2.sysu.edu.cn), Sun Yat-sen University


[Deeplab_v3_paper]: https://arxiv.org/abs/1706.05587
[Deeplab_v3p_paper]: https://arxiv.org/abs/1802.02611
[FCN_paper]: https://arxiv.org/abs/1411.4038
[SegNet_paper]: https://arxiv.org/abs/1511.00561
[DenseASPP_paper]: http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html
[RefineNet_paper]: https://arxiv.org/abs/1611.06612
[PSPNet_paper]: https://arxiv.org/abs/1612.01105


