## System Requirement

ScLNet is a CNN backbone designed for viral perform region segmentation and extract regional contours simultaneously from OCT images with sclera lens on eyes with regular and irregular cornea . 



![image-20231025103251740](https://github.com/smallJude/ScLNet/blob/main/image/image_framework.tif)



According to segmentation result ,we can extract the quantitative parameters for evaluating the lens fitting. 

![image-20231025101503269](https://github.com/smallJude/ScLNet/blob/main/image/image2.tif)

### 

## System Requirement

#### Software Requirements

##### Os  Requirements

can run on windows. The package has been tested on the Windows systems:

##### Python Dependencies

```txt
gast==0.3.3
grpcio==1.30.0
h5py==2.10.0
idna==2.10
image==1.5.33
imageio==2.9.0
imgaug==0.4.0
importlib-metadata==1.7.0
importlib-resources==5.4.0
natsort==7.1.0
nets==0.0.3.1
networkx==2.5
numpy==1.16.0
opencv-python==4.1.2.30
pandas==1.1.5
Pillow==7.0.0
six==1.15.0
scikit-learn==0.23.2
scipy==1.5.2
sklearn==0.0
sqlparse==0.4.2
termcolor==1.1.0
tflearn==0.3.2
threadpoolctl==2.1.0
tqdm==4.64.1
```



## Installation Guide

#### Install dependency packages

1. Install `python 3.6` following the [official guide](https://www.python.org/).

2. Install `tensorflow` following the [official guide](https://www.tensorflow.org/?hl=zh-cn).

3. Install other dependencies:

   ```
   pip install -r requirements.txt
   ```

#### Install RESEPT from GitHub

```txt
git clone https://github.com/smallJude/ScLNet
cd ScLNet
```



## Data preparation

#### Data

Input image :OCT image and its manually annotated  semantically segmented label image ,meanwhile train.py  can automatically generate labels for image boundaries based on labels.

#### Segmentation model file

[model_file.txt](https://github.com/smallJude/ScLNet/blob/main/model_file.txt)  It is a pre-trained segmentation model file in the hd5 format, which should be provided in predicting  OCT images. 



## Running Training Procedures

You can train the network with any dataset you like. 

```
python train.py
```

## Running Prediction Procedures

The predict.py script can handles prediction for individual images. Prediction requires trained weights for a given backbone. We have provided weights and precompiled models for ScLNet, which can be downloaded using the script in the [model_file.txt](https://github.com/smallJude/ScLNet/blob/main/model_file.txt) directory

```
python predict.py
```

