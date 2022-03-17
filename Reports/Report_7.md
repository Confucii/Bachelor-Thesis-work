
## Task 6
### Previous work fix DenseMNIST
#### Changes
![Image](../Images/Fix_1.png "Fix 1")
##### Old
![Image](../Images/Fix_1_Old.png "Fix 1 old")
##### New
![Image](../Images/Fix_1_New.png "Fix 1 new")
![Image](../Images/Fix_2.png "Fix 2")
##### Old
![Image](../Images/Fix_2_Old.png "Fix 2 old")
##### New
![Image](../Images/Fix_2_New.png "Fix 2 new")
#### Result comparison
The results with changes are usually a bit worse than the initial version, but as hot-one encoded vectors are not used, they new versions work faster.
##### Old
![Image](../Images/Result_MNIST_Old.PNG "Results old")
##### New
![Image](../Images/Result_MNIST_New.PNG "Results new")

### Previous work fix LFW
#### Changes
![Image](../Images/Fix_3.png "Fix 3")
##### Old
![Image](../Images/Fix_3_Old.png "Fix 3 old")
![Image](../Images/Fix_3_Old_1.png "Fix 3 old")
##### New
![Image](../Images/Fix_3_New.png "Fix 3 new")
![Image](../Images/Fix_3_New_1.png "Fix 3 new")
![Image](../Images/Fix_2.png "Fix 2")
##### Old
![Image](../Images/Fix_2_Old.png "Fix 2 old")
##### New
![Image](../Images/Fix_2_New.png "Fix 2 new")
#### Result comparison
The results are even more interesting
##### Old
![Image](../Images/Result_LFW_Old.PNG "Results old")
##### New
![Image](../Images/Result_LFW_New.PNG "Results new")

### Focal loss
I have decided to use pre-trained vgg11 and Adam optimizer. I have been testing the model on Fashion MNIST dataset. However the results are quite odd.
#### Focal loss implementation
![Image](../Images/Focal_Loss_Module.png "Focal loss implementation")

#### Results
The oddity comes from results, they are inconsistent and I can't find the dependancy. 
##### Sometimes model works like this:
![Image](../Images/Focal_Loss.png "Focal loss results")
##### Sometimes this happens:
![Image](../Images/focal_loss_strange.png "Focal loss strange results")
##### Debugging yields this
![Image](../Images/Focal_Loss_debugging_oddity.png "Focal loss strange results")