# Model Overview
This repository contains the code for NNDA-UNETR. NNDA-UNETR is the state-of-the-art on Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset.

# Installing Dependencies
Dependencies can be installed using:
``` bash
conda create -n btcv python=3.8
cd BTCV
source activate btcv
pip install -r requirements.txt
```

# Data Preparation
<!-- ![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0) -->

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4.Gallbladder 5.Esophagus 6. Liver 7. Stomach 8.Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12.Right adrenal gland 13.Left adrenal gland.
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)

Please download the json file from this link.

We provide the json file that is used to train our models in the following <a href="https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing"> link</a>.

Once the json file is downloaded, please place it in the same folder as the dataset. Note that you need to provide the location of your dataset directory by using ```--data_dir```.

# Training

A NNDA-UNETR network with standard hyper-parameters for multi-organ semantic segmentation (BTCV dataset) is be defined as:

``` bash
model = NNDA_UNETR(in_channels=1, out_channels=14, img_size=(96, 96, 96), feature_size=16, num_heads=4,
		   norm_name = 'batch', depths = [3,3,3,3], dims = [32, 64, 128, 256], do_ds = False)
```


The above NNDA-UNETR model is used for CT images (1-channel input) with input image size ```(96, 96, 96)``` and for ```14``` class segmentation outputs.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:
``` bash
python main.py 
--feature_size=48  
--resume_ckpt 
--use_checkpoint  
--save_checkpoint 
--optim_name=Lion
```

# Evaluation and Segmentation Output

To evaluate a `NNDA-UNETR` on a single GPU, place the model checkpoint in `pretrained_models` folder and
provide its name using `--pretrained_model_name`:

```bash
python test.py   
--infer_overlap=0.5 
--pretrained_model_name='model.pt' 
--pretrained_dir='./runs/test/'
```

