# Installation

Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. This repository should be used with [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [mmsegmentation==0.12.0](https://github.com/open-mmlab/mmsegmentation/releases/tag/v0.12.0), and [cyanure](https://github.com/jmairal/cyanure) for evaluation on downstream tasks. To get the full dependencies, please run:

```
pip3 install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.3.9
pip3 install pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops

# install apex
pip3 install git+https://github.com/NVIDIA/apex \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

# install mmdetection for object detection & instance segmentation
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip3 install -r requirements/build.txt
pip3 install -v -e .
cd ..


# install mmsegmentation==0.12.0 for semantic segmentation
git clone -b v0.12.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip3 install -v -e .
cd ..

# install cyanure-mkl for logistic regression
pip3 install mkl
git clone https://github.com/jmairal/cyanure.git
cd cyanure
sudo python3 setup_cyanure_mkl.py install
cd ..
```

## How to run this on PathAI cluster later version
Initialize a virtual environment like `contrimix_test_5`, run the following commands
```
conda create -n ibot_env python=3.6 -y
conda activate ibot_env

# Install Pytorch, cudatool kit and torch vision - compatible with cuda 11.7
conda install pytorch-gpu==1.7.1 cudatoolkit torchvision==0.8.2 -c pytorch

# Install cudatoolkit
conda install -c conda-forge cudatoolkit-dev 

# Install mmcv: version 1.2.4 is needed for the mmdetection 2.10.0, see https://mmdetection.readthedocs.io/en/v2.10.0/get_started.html#installation
git clone -b v1.2.4 https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements.txt
pip install -e . -v

# Install other packages.
pip3 install pytest-runner scipy tensorboardX scikit-learn DALL-E
conda install pytorch::faiss-gpu 
conda install tqdm lmdb pyarrow==2.0.0 timm munkres six einops packaging s3fs

# Build apex - See the instruction [here](https://github.com/NVIDIA/apex).
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# See installation for mmdetection [here](https://mmdetection.readthedocs.io/en/latest/get_started.html)
# For the mmcv detection dependency, see https://mmdetection.readthedocs.io/en/v2.10.0/get_started.html
# Install mmdetection - ok
git clone -b v2.1.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

# install mmsegmentation==0.12.0 for semantic segmentation - ok
git clone -b v0.12.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip3 install -v -e .
cd ..

git clone https://github.com/thnguyn2/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip3 install -r requirements/build.txt
pip3 install -v -e .
cd ..

# install cyanure-mkl for logistic regression
conda install mkl cyanure
```