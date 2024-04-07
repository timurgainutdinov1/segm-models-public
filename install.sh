#!/bin/bash

python3.9 -m venv segm_models
source segm_models/bin/activate

python -m pip install --upgrade pip
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_toolbelt==0.6.3 --no-deps
pip install segmentation_models_pytorch==0.3.3 --no-deps
pip install pretrainedmodels==0.7.4 --no-deps
pip install timm==0.9.2 --no-deps
pip install efficientnet-pytorch==0.7.1 numpy==1.26.4 opencv-python==4.9.0.80 albumentations==1.4.2 tqdm==4.66.2 six==1.16.0 tensorboard==2.16.2 matplotlib==3.8.3

mkdir "logs"
chmod +x train.sh
chmod +x validate.sh