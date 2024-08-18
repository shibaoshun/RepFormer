# RepFormer: Rain streak and image edge prompting Transformer for single image deraining
Baoshun Shi，Wei Ma，Shengnan Yan
## Abatract
Recent deep learning-based single image deraining (SID) methods have achieved remarkable results, but they either utilize Transformer in rain streak and non-streak regions equally or work locally ignoring global information, suffering from incomplete restoration of fine details in rain streaks and global edges. To address this, we propose RepFormer, a novel rain streak and image edge prompting Transformer-based SID network, which focus on restoring contents in rain streaks and preserving image edges. Specifically, we elaborate a rain streak estimation network and leverage the segment anything model (SAM) to estimate rain streaks and image edges, respectively. We inject these high-frequency features into the Transformer architecture via a prompt embedding module, and propose a rain streak and image edge prompting Transformer, which utilizes the non-local information to restore the fine details in the rain streaks and preserve image edges. Comprehensive experiments on various representative datasets validate the superiority of our method.

![image name](https://github.com/mawei-north/RepFormer/blob/7c32dc7e07c0647375459f8fc821417b57cacdb0/figss/RepFormer.png)
## Installation
The model is built in PyTorch 1.10.0 and  trained with NVIDIA 2080Ti GPU.
For installing, follow these intructions
```
conda create -n pytorch1 python=3.8
conda activate pytorch1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```
Install warmup scheduler
```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```
## Training and Testing
Training and Testing codes for RSEN, SAM and RepFormer are provided in their respective directories.
## Dataset
We train and test our RepFormer in Rain100L, Rain200L. The download links of datasets are provided.
+ Rain100L: 200 training pairs and 100 testing pairs. Download from [Datasets](https://pan.baidu.com/s/16n5hKHkr2rKlz2kBlI5JSQ?pwd=wxdm)
+ Rain200L: 1800 training pairs and 200 testing pairs. Download from [Datasets](https://pan.baidu.com/s/16n5hKHkr2rKlz2kBlI5JSQ?pwd=wxdm)
## Pre-trained Models
### For RSEN
Please download checkpoints from [RSEN](https://pan.baidu.com/s/1VyZRqqfCUSZm5zilCIlw9g?pwd=edij)
### For SAM
Please download checkpoints  for the corresponding model type from [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
### For RepFormer
Please download checkpoints from


## Performance Evaluation 
![](https://github.com/mawei-north/RepFormer/blob/4709e59d3aafe69bee39395bb84781d8c51f6469/figss/results.png)
![]()
