**BREAST CANCER CLASSIFICATION using Pytorch**

**Description**: Given screening mammograms obtained from experts, the trained model has to predict whether or not the test mamograms contains cancer.

**Dataset**: There are 2 datasets in this project
1) CBIS-DDSM (Curated Breast Imaging Subset of DDSM) includes 2,620 scanned film mammography studies (JPEG format)
2) 2023 RSNA Screening Mammography Breast Cancer Detection AI Challenge includes over 50000 images (dicom format)

For CBIS-DDSM, only full size images are used, croped images are ignored. For RSNA AI Challenge dataset, total 6000 images are used. The combined dataset from two sources is split randomly into 70% train set and 30% test set. 

**Data Augmentation**: Some data augmentation method likes RandomResizedCrop, ElasticTransform, RandomAutocontrast applied for train set only. Note that, some previous analysis show that data in RSNA AI Challenge dataset are imblanced classes and no there no sampling method or MixUp method applied to the dataset.


**DNN Model**: In the list of model for experiments includes ResNet50, DensNet, EfficientNent, ConvNext, the last one outperformed other models. When setup training for ConvNet model, a finite number of ConvNext blocks are freezed for fintuning from pretrained model on IMAGENET1K

**Training model**: Training proceduce is setup in TPU environment of Kaggle server. The model and data are distributed across 8 cores of TPU. Accuracy and loss values in each process are collected and take avergage by using xm.all_reduce() for all groups.

**Training results** 

![](Results\accuracy.JPG)
![](Results\loss.JPG)
