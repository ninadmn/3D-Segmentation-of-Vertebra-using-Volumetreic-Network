# 3D-Instance-segmentation-of-Vertebra-using-Mask-RCNN
Thesis project on 3D instance segmentation of vertebra using Mask RCNN


# 3D Segmentation of Vertebra using Volumetric Network
> This is an example of the CT images of vertebra segmentation from spineweb


## Prerequisities
The following dependencies are needed:
- numpy 
- SimpleITK 
- opencv-python 
- tensorflow
- keras
- pandas 
- scikit-learn 
- matplotlib
- plotly

## How to Use

**1、Preprocess**

**Vertebra Detection**

* analyze the ct image,and get the slice thickness and window width and position along with generating 3D interactive plot and extraction of masks:run the GenerateMasks.ipynb
* generate vertebra ct image and mask:run the get2dScans.py
* generate patch(128,128,128) of vertebra image and mask:run the preparePatches.py
* save vertebra data and mask into csv file run the saveDetailsToCSV.py



**2、Segmentation of Vertebra**
* the VNet model

![](VNetDiagram.png) 

* To train and generate the result run VNet.py

