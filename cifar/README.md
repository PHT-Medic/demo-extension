# Cifar 10 example train
Runs one epoch of training on the cifar10 dataset using a simple CNN model. The model is trained on the GPU if available 
if a previous checkpoint exists it is loaded and training is continued. The model is saved after each epoch.

This requires torch and torchvision to be installed. The examples are based on the [PyTorch CIFAR10 example](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)


## Downloading the data
Install requests and download and extract the data:
```shell
pip3 install requests
```
tar contents get extracted to `data/cifar-10-batches-py` folder
```shell
python get_data.py
```