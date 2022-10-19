import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CHECKPOINT_PATH = '/opt/pht_results/checkpoint.pth'

DATA_PATH = '/opt/train_data'

BATCH_SIZE = 4


def load_data():
    print("Loading data...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    print("Data loaded.")

    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cifar_train():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load checkpoint if it exists
    if os.path.isfile(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("Previous loss: ", loss)
        print("Checkpoint loaded.")

    trainloader, testloader = load_data()

    print("Starting training...")
    # run a single epoch over the data
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[1, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print(f'Finished Training - Running Loss: {running_loss}')

    # check that checkpoint path exists
    if not os.path.exists(os.path.dirname(CHECKPOINT_PATH)):
        print("Checkpoint path does not exist, creating...")
        os.makedirs(os.path.dirname(CHECKPOINT_PATH))

    # save checkpoint
    print('Saving checkpoint...')
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
    }, CHECKPOINT_PATH)


if __name__ == '__main__':
    cifar_train()
