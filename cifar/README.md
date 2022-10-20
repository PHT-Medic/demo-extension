# Cifar 10 example train
Runs one epoch of training on the cifar10 dataset using a simple CNN model. The model is trained on the GPU if available 
if a previous checkpoint exists it is loaded and training is continued. The model is saved after each epoch.

This requires torch and torchvision to be installed inside the image. The examples are based on the [PyTorch CIFAR10 example](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Downloading the data
Install requests and download and extract the data:
```shell
pip3 install requests
```
tar contents get extracted to `data/cifar-10-batches-py` folder
```shell
python get_data.py
```

## Running the train
After creating a train with the `cifar_train.py` script as entrypoint adapt the following command to
the stations name space and the train id.
This command will mount the previously downloaded data into the container and run the train.

```json
{
  "repository": "<REGISTRY>/<STATION-REPO>/<TRAIN-ID>",
  "tag": "latest",
  "volumes": {
    "/home/ubuntu/station/station_data": {
        "bind": "/opt/train_data",
        "mode": "ro"
    }
  },
  "gpus": "all"
}
```
