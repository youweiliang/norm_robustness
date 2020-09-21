# The L1 and L_inf Norms of CNN Layers and Adversarial Robustness
This repository hosts the python code for our paper [Large Norms of CNN Layers Do Not Hurt Adversarial Robustness](https://arxiv.org/abs/2009.08435). 

## Prerequisites
* The code has been tested with Python 3.6.
* To avoid breaking the packages in your python environment, we recommend creating a virtual environment for this project. Then activate the virtual environment and run `pip install -r requirements.txt` to install the required packages. 

## Usage of The Norm Decay Algorithm
Our norm decay algorithm is very easy to use and is very efficient. Currently, it only supports PyTorch models. 
Suppose we have a PyTorch model `model` (subclass of `nn.Module`). To use norm decay in the training of the model, we first bind some methods to the model
```python
from lip.add_lip import bind_lip
bind_lip(model, norm='1-norm', beta=1e-3)
```
where `norm` can be `'1-norm'` or `'inf-norm'` and `beta` is the regularization parameter.  
Then, before each update of the model parameter (i.e., before `optimizer.step()`), apply the norm decay 
```python
lipc, all_lip = model.add_lip_grad(linear=True, conv=True, bn=False)
```
where `linear, conv, bn` controls whether norm decay is applied to fully connected layers, convolutional layers, and batch normalization (BN) layers, respectively. The return `all_lip` is a list of the norms of all layers in the model and `lipc` is simply `sum(all_lip)`. There is also a method solely for calculating the norms:
```python
lipc, all_lip = model.calc_lip()
```
Since we find that it is quite difficult to control the norms of BN with norm decay, we also provide a method for projecting the norms of BN to a fixed value using the code 
```python
model.project_bn(proj_to=5)
``` 

### Note  
* The norm decay algorithms and related code are located in the directory `./lip`. _**In the code**_, "lip" is used as a synonym of "norm" (though they bear different meanings in the paper and literature). 
* Most of the code is based on PyTorch and only singular value clipping (SVC) is based on TensorFlow because SVC requires singular value decomposition for _complex matrices_ which is not available in PyTorch. 

## Experiments
These are the experiments conducted in our paper. 
### Algorithmic Efficiency Comparison of Computing Norms of Convolutional Layers
Run the command `python speed_test.py`. The results will be printed on the console. 

### Norm Regularization
Run the command `bash run_norm_regularization.sh`. The clean and robust accuracy will be saved to `./regularization/acc_table.txt`. The images of the norms of fully connected and convolutional layers will be saved to `./regularization/log/{model}/conv_linear` and the norms of batch normalization layers to `./regularization/log/{model}/bn`, where `{model}` could be `vgg`, `resnet`, `senet`, and `regnet`.

### Norm Comparison of Adversarially Robust Models and Non-Adversarially Robust Models
Run the command `bash run_adv_training.sh`. The images of the distribution of norms will be saved to `./img_den`, and the images of comparison between norms for individual layers will be saved to `./img_compare_norm`. 

## Acknowledgements
Part of the code is based on these awesome GitHub repositories: [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), [AT_HE](https://github.com/ShawnXYang/AT_HE), and [auto-attack](https://github.com/fra31/auto-attack). 
