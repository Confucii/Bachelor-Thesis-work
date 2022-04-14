
## Report 9
### Theoretical
9.5 pages written
#
Google Drive link: [Link](https://drive.google.com/drive/u/0/folders/1z9pRSUUqAFRH-cxjEwzO74S6yV0WnkjA "GoogleDrive")
### Practical
Files for CIFAR, LFW and Fashion MNIST with CCE(not weighted), focal loss are underdeveloped: Cannot apply stratify; unsure if datasets are preprocessed correctly; __cannot understand if I need weighted term for focal loss and how to calculate weighted cross entropy (specifically how to get class counts from datasets).__
### CCE (Fashion)
![Image](../Images/cce_fashion_9.png "CCE Fashion")
#
Test accuracy cannot increase higher than 0.75. Test loss is getting higher.
### Focal loss (Fashion)
![Image](../Images/focal_fashion_9.png "Focal Fashion")
#
Test accuracy cannot increase higher than 0.75. Test loss is getting higher.
### CCE (LFW)
![Image](../Images/cce_LFW_9.png "CCE LFW")
# 
Test accuracy is horrible. Not stratified? Overfitting?
### Focal loss (LFW)
![Image](../Images/focal_LFW_9.png "Fashion LFW")
#
Test accuracy is horrible. Not stratified? Overfitting?
### CCE (CIFAR)
![Image](../Images/cce_cifar_9.png "CCE CIFAR")
### Focal loss (CIFAR)
![Image](../Images/focal_cifar_9.png "Focal CIFAR")
### Stratify 
![Image](../Images/stratify.png "Focal CIFAR")
![Image](../Images/stratify_error.png "Focal CIFAR")
