## The L1 and L_inf Norms of CNN Layers and Adversarial Robustness
This repository hosts the python code for our paper [Large Norms of CNN Layers Do Not Hurt Adversarial Robustness](https://arxiv.org/abs/2009.08435). 

### Prerequisites
* The code has been tested with Python 3.6.
* Create a virtual environment, activate the virtual environment, and run `pip install -r requirements.txt`.

### Timing
Run the command `python speed_test.py`. The results will be printed on the console. 

### Norm Regularization
Run the command `bash run_norm_regularization.sh`. The clean and robust accuracy will be saved to `./regularization/acc_table.txt`. The images of the norms of fully connected and convolutional layers will be saved to `./regularization/log/{model}/conv_linear` and the norms of batch normalization layers to `./regularization/log/{model}/bn`, where `{model}` could be `vgg`, `resnet`, `senet`, and `regnet`.

### Norm Comparison of Adversarially Robust Models and Non-Adversarially Robust Models
Run the command `bash run_adv_training.sh`. The images of the distribution of norms will be saved to `./img_den`, and the images of comparison of individual norms will be saved to `./img_compare_norm`. 

### Note  
* The norm decay algorithms and related code are located in the directory `./lip`. 
* Most of the code is based on PyTorch and only singular value clipping (SVC) is based on TensorFlow because SVC requires _complex_ SVD (singular value decomposition) and inverse FFT (fast fourier transform) which are not available in PyTorch. 

### Acknowledgement
Part of the code is based on these GitHub repositories [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), [AT_HE](https://github.com/ShawnXYang/AT_HE), and [auto-attack](https://github.com/fra31/auto-attack). 
