FROM nvcr.io/nvidia/pytorch:22.09-py3

COPY ./cifar_train.py /opt/pht_train/cifar_train.py

WORKDIR /opt/pht_train
CMD ["python", "cifar_train.py"]
