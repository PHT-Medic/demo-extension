import requests
import tarfile
import io


def download_cifar_10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    r = requests.get(url, allow_redirects=True)
    tar = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz")
    tar.extractall(path="./data")


if __name__ == '__main__':
    download_cifar_10()
