# Final Project for Deep Learning 02456

This project aims to re-implement the Denoising Difussion Probabilistic Models (DDPM), training and testing it on the MNIST and CIFAR-10 datasets.


## Contents

This repository contains 4 files and several folders, which are all essential for this project and can be seen as proof of implementation. Due to the nature of our cooperation, there are two implementations of DDPM, corresponding to each dataset, which will be written in separate jupyter notebooks.

* DDPM_MNIST.ipynb: 

    The main file for MNIST dataset, including model definition and training loop. 

* FID_Calculation_MNIST.ipynb: 

    The main file to calculate FID score for trained data (trained by DDPM_MNIST.ipynb) also includes generating a series of images for display. 

* DDPM_CIFAR.ipynb: 

    The main file for MNIST dataset, including model definition and training loop. 

* FID_Calculation_CIFAR.ipynb: 

    The main file to calculate FID score for trained data (trained by DDPM_CIFAR.ipynb) also includes generating a series of images for display. 


## Warning:

* If you run each jupyter notebook, then several images will be saved, and cover others generated earlier. 

* The path to dataset may need to be adjusted since we didn't upload the dataset files. (too big to upload to github)

* The path to model may not be available since we didn't upload the trained model files (*.pt), but the results can be seen in the report. 


## Members of this group

* Shu-Han Chien, s222458

* Hanqing Cao, s230037

* Hao Zhang, s222375

* Yichen Wang, s222406
