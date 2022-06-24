# Computed-Tomography-Liver-Segmentation-and-Classification
My graduation project in Nanjing University
## Introduction
This is an auxiliary diagnosis system that executes liver segmentation first and then executes classification. Previous students found that compared with manually acquired liver segments, using automatically acquired liver segments will produce a negative impact on classification. Typically, people treat the last feature map with the Sigmoid function. Then, set a threshold to get binary masks of the liver, which is named binarization operation here. I found that if I remove the binarization operation and then got a kind of liver mask named Prob-Mask, I will ameliorate the negative influence of the automatically acquired liver segments. Although joint learning has been proposed for a while, it is interested to find that.
### System Overview Attach with Code
![outline](https://user-images.githubusercontent.com/108105092/175440357-4f5fd8ec-b24e-44e1-808b-9f5e2cc8f84f.png)

This is the overview of the whole system. You can match the codes with this outline.

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
    
│   ├── train.py
│   ├── eval.py
│   └── classification
    │   ├── train_cls.py
        └── test_cls.py
      
```
