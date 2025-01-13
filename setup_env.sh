#!/bin/bash
# 设置conda镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
# 设置搜索时显示通道地址 从channel中安装包时显示channel的url，这样就可以知道包的安装来源
conda config --set show_channel_urls yes

# Create a new conda environment
# conda create -y -n latentsync python=3.10.13
conda activate latentsync

# # Install ffmpeg
# conda install -y -c conda-forge ffmpeg

# # Python dependencies
# pip install -r requirements.txt

# OpenCV dependencies
apt -y install libgl1
echo 'huggingface 安装'
export HF_ENDPOINT="https://hf-mirror.com" 
# Download all the checkpoints from HuggingFace
huggingface-cli download chunyu-li/LatentSync --local-dir checkpoints --exclude "*.git*" "README.md"

# Soft links for the auxiliary models
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
