# NNDA-UNETR: A Plug-and-Play Parallel Dual Attention Block in U-Net with Enhanced Residual Blocks for Medical Image Segmentation
---
## Installation
#### 1. System requirements
We run NNDA-UNETR on a system running Ubuntu 22.04, with Python 3.6, PyTorch 1.8.1, and CUDA 11.7. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation. Thus, systems lacking a suitable GPU would likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 2080 TI GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2. Installation guide
We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
cd ACDC
conda env create -f environment.yml
source activate nnFormer
pip install -e .
```

#### 3. Functions of scripts and folders
- **For evaluation:**
  - ``NNDA-UNETR-main/ACDC/NNDA/inference_acdc.py``
  
- **Data split:**
  - ``NNDA-UNETR-main/ACDC/NNDA/dataset_json/``
  
- **For inference:**
  - ``NNDA-UNETR-main/ACDC/NNDA/inference/predict_simple.py``
  
- **Network architecture:**
  - ``NNDA-UNETR-main/ACDC/NNDA/training/network_training/nnFormerTrainerV2_nnformer_acdc.py``
  
- **For training:**
  - ``nnFormer/nnformer/run/run_training.py``
  
- **Trainer for dataset: ACDC **
  - ``ACDC/NNDA/training/network_training/nnFormerTrainerV2_nnformer_acdc.py``
---

## Training
#### 1. Dataset download
Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)


The split of ACDC dataset is available in ``ACDC/NNDA/dataset_json/``.

#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./NNDA/
./DATASET/
  ├── nnFormer_raw/
      ├── nnFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── nnFormer_cropped_data/
  ├── nnFormer_trained_models/
  ├── nnFormer_preprocessed/
```
You can refer to ``ACDC/NNDA/dataset_json/`` for data split.

After that, you can preprocess the above data using following commands:
```
nnFormer_convert_decathlon_task -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task001_ACDC


nnFormer_plan_and_preprocess -t 1 --verify_dataset_integrity

```

#### 3. Training and Testing
- Commands for training and testing:

```
nnFormer_train 3d_fullres nnFormerTrainerV2_nnformer_acdc  1  0
```
If you want use your own data, please create a new trainer file in the path ```ACDC/NNDA/training/network_training``` and make sure the class name in the trainer file is the same as the trainer file. Some hyperparameters could be adjust in the trainer file, but the batch size and crop size should be adjust in the file```ACDC/NNDA/run/default_configuration.py```.
 
- You can download our pretrained model weights via this [link](https://drive.google.com/drive/folders/1yvqlkeRq1qr5RxH-EzFyZEFsJsGFEc78?usp=sharing). Then, you can put model weights and their associated files in corresponding directories. For instance, on ACDC dataset, they should be like this:
- We have placed the pretrained model weights in the following path.

```
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model
../DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/fold_0/model_best.model.pkl
```

- Commands for inference
```
nnFormer_predict -i ../DATASET/nnFormer_raw/nnFormer_raw_data/Task001_ACDC/imagesTs -o /DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/inferTs -t 1 -m 3d_fullres -f 0 -chk model_best -tr nnFormerTrainerV2_nnformer_acdc
cd ACDC/NNDA
python inference_acdc.py 0
```

#### 4. Visualization Results

You can download the visualization results of nnFormer, UNETR++ and NNDA-UNETR from this [link](https://drive.google.com/file/d/1Lb4rIkwIpuJS3tomBiKl7FBtNF2dv_6M/view?usp=sharing).

