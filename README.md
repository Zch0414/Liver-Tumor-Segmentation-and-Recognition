# Swin-Ynet: Joint Learning for Liver Tumor Segmentation and Classification with a Swin-Transformer

This is the repository of my EECS 504 project with help from Yifan Wang in the experiment part.

----

## Two-stages Liver Tumor Recognition

This work draws inspiration from the Y-Net but differs in its primary objective. While the Y-Net focuses mainly on improving segmentation results for breast biopsy images, my aim is to address the adverse impact of segmentation model outputs on classification models, as observed in the [Two-stage Liver Tumor Recognition](https://github.com/Zch0414/Liver-Tumor-Segmentation-and-Recognition/tree/2stage). To achieve this, I introduce a patch-based classification component between the encoder and decoder of a U-shaped model. Prior to entering the decoder, the classification component predicts a grid-like region of interest. I hypothesize that this approach will foster mutual benefits between the classification and segmentation components.

![swin](https://github.com/Zch0414/Liver-Tumor-Segmentation-and-Recognition/blob/swin-ynet/img/pipeline.png)

----

## Result

Classification Accuracy: 3% up 

Liver Dice Score: 1% up

Liver Tumor Dice Score: 5% up






