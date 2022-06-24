# Computed-Tomography-Liver-Segmentation-and-Classification
My graduation project in Nanjing University
## Introduction
This is an auxiliary diagnosis system that executes liver segmentation first and then executes classification. Previous students found that compared with manually acquired liver segments, using automatically acquired liver segments will produce a negative impact on classification. Typically, people treat the last feature map with the Sigmoid function. Then, set a threshold to get binary masks of the liver, which is named binarization operation here. I found that if I remove the binarization operation and then got a kind of liver mask named Prob-Mask, I will ameliorate the negative influence of the automatically acquired liver segments. Although joint learning has been proposed for a while, it is interested to find that. The whole project is based on the LiTS dataset.
## Code Directory
```bash
├── LITS17
    └── *.nii.zip
├── main
│   ├── Extract.py
    └── data
        └── *.nii
        
│   ├── data_process.py
    └── data_test
    |   └── *.png
    └── data_train
        └── *.png
        
│   ├── train.py
│   ├── eval.py

│   ├── data_process_cls.py
    └── data_cls
    |   ├── image
    |       └── *.png
    |   ├── label
    |       └── *.npy
    |   ├── test
    |       └── *.npy
    |   ├── train
            └── *.npy
    
│   ├── utils
    |   ├── dataset.py
    |   ├── dataset_cls.py
    |   └── dice_loss.py
    
│   ├── unet
    |   ├── unet_model.py
    |   ├── unet_part.py
    |   └── resnet.py
    
│   └── classification
    │   ├── train_cls.py
        └── test_cls.py      
```
## System Overview Attach with Code
![outline](https://user-images.githubusercontent.com/108105092/175440357-4f5fd8ec-b24e-44e1-808b-9f5e2cc8f84f.png)
To make it more acceptable for the readers, I provide the overview of the system. Firstly, run Extract.py and data_process.py to get image data for the segmentation network, which is based on the U-Net. In data_process.py, I refer to https://github.com/assassint2017/MICCAI-LITS2017. You will acquire the image on the very left of the overview after the first step. Secondly, you can run train.py to train the segmentation network, the parameters of the network will be saved, which is not shown in the directory. Thirdly, run data_process_cls.py to get the data for the classification network. This step matches with the ProcessorI&II of the overview. Lastly, you can change the directory to classification and run train_cls.py to train the classification network, the parameters of the network will be saved too, which is not shoen in the directory as well.

## Result
For the segmentation task, the network can get 90.98% on dice coefficient, which can match to the result in some publications. For the classification task, I have to admit the result is not very stable, which can be indicated in the following picture.
![result_classification](https://user-images.githubusercontent.com/108105092/175536323-49c711f8-b647-4fb7-bd11-b78d8ff2a381.png)



