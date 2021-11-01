
# Automatic Correction for Handwritten Mathemetical Formula 
Detecting lines of mathematical problems then solve it!
## We develop two kinds of model for recognizing math operators and numbers.
**Low-level model**: Adpoting BOW + SVM for image classification

**CNN-based model**: LeNet trained on MNIST

## Pipeline is shown below
![image](https://user-images.githubusercontent.com/72722062/130405791-5451b443-74bf-40bc-a11d-d8bf45e58488.png)

## To improve the accuracy, we extact the skeleton for each element detected.
<img width="600" alt="Screen Shot 2021-08-23 at 3 23 40 PM" src="https://user-images.githubusercontent.com/72722062/130407003-1efa36f8-30e6-4942-970a-570485a2b491.png">


* Dataset: CROHME of ICFHR14 and MNIST
* Environment: Tensorflow 2.4, Opencv 4.5 and python 3.7+


