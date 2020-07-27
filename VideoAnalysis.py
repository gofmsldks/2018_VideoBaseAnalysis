mport torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.autograd import variable
import cv2

input_size =784
hidden_size = 50
num_classes = 10
num_epochs = 1
batch_size = 1
learning_rate = 0.001

model = MyNet(input_size, hidden-size,num_classes)
loss_func = MyLoss()
class Dataset:
    def __init__(self, folder_path):
        self.n_classes = num_classes
        self.all_train_img_list = []
        self.all_train_label_list = []

        for i in range(self.n_classes):
            self.train_list = os.listdir(folder_path=str(i))
            self.train_list = [folder_path+str(i)+'/'+file_name for file_name in self.train_list]
            self.all_train_img_list = self.all_train_img_list+self.train_list
            self.all_train_img_list = self.all_train_label_list+[i]*len(self.train_list)

    def __getitem__(self, idx):
        img = cv2.inread(self.all_train_label_list[idx], cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        img = (img/255.).astype(np.float32)
        label = self.all_train_label_list[idx]
        label = np.eye(self.n_classes)[label].astype(np.float32)
        return img, label

    def __len__(self):
        return  len(self.all_train_img_list)

class MyNet(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(MyNet, self).__intit__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLu()
        self.fc2 = nn.Linear(hidden_size, num-classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, y, y_pred):
        return ((y-y_pred)**2).mean()

if __name__ = '__main__':
    train_dataset = Dataset(folder_path='')
    train_data_loader = DataLoader(train_dataset, atch_size = batch_size, shuffle = False, num_workers=8)
    test_dataset = Dataset(folder_path='')
    test_train_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle= True, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters90, Ir = learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):

            images  = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(labels, outputs)
            loss.backward()
            optimizer.step()

            if(i + 1) % 100 ==0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset)// batch_size, loss.data[0]))

                correct = 0
                total = 0

                for i, (images, labels) in enumerate(test_data_loader):
                    images = Variable(images.view(-1, 28 * 28))
                    labels = Variable(labels)
                    labels_y = torch.max(labels.data, 1)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (labels_y[1] == predicted).sum()
                print('Accuracy of the network on the 10k test images: %d %%' % (100 * correct / total))

————————————————————————————————————————————

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.autograd import variable
import cv2

input_size =784
hidden_size = 50
num_classes = 10
num_epochs = 1
batch_size = 1
learning_rate = 0.001

model = MyNet(input_size, hidden-size,num_classes)
loss_func = MyLoss()
class Dataset:
    def __init__(self, folder_path):
        self.n_classes = num_classes
        self.all_train_img_list = []
        self.all_train_label_list = []

        for i in range(self.n_classes):
            self.train_list = os.listdir(folder_path=str(i))
            self.train_list = [folder_path+str(i)+'/'+file_name for file_name in self.train_list]
            self.all_train_img_list = self.all_train_img_list+self.train_list
            self.all_train_img_list = self.all_train_label_list+[i]*len(self.train_list)

    def __getitem__(self, idx):
        img = cv2.inread(self.all_train_label_list[idx], cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        img = (img/255.).astype(np.float32)
        label = self.all_train_label_list[idx]
        label = np.eye(self.n_classes)[label].astype(np.float32)
        return img, label

    def __len__(self):
        return  len(self.all_train_img_list)

class MyNet(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(MyNet, self).__intit__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLu()
        self.fc2 = nn.Linear(hidden_size, num-classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, y, y_pred):
        return ((y-y_pred)**2).mean()

if __name__ = '__main__':
    train_dataset = Dataset(folder_path='')
    train_data_loader = DataLoader(train_dataset, atch_size = batch_size, shuffle = False, num_workers=8)
    test_dataset = Dataset(folder_path='')
    test_train_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle= True, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters90, Ir = learning_rate)






    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):

            images  = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(labels, outputs)
            loss.backward()
            optimizer.step()

            if(i + 1) % 100 ==0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset)// batch_size, loss.data[0]))


correct = 0
total = 0

for i, (images, labels) in enumerate(test_data_loader):
    images = Variable(images.view(-1, 28 * 28))
    labels = Variable(labels)
    labels_y = torch.max(labels.data, 1)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (labels_y[1] == predicted).sum()
    print('Accuracy of the network on the 10k test images: %d %%' % (100 * correct / total))




————————————————————————————————————————————
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        #print(x.shape)
        x = x.view(-1,16*5*5)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return F.softmax(x, -1)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, y_pred, y):
        return ((y - y_pred)**2).mean()

class Dataset:
    def __init__(self, folder_path):
        self.n_classes = 10
        self.all_train_img_list = []
        self.all_train_label_list = []

        for i in range(self.n_classes):
            train_list = os.listdir(folder_path + str(i))
            train_list = [folder_path + str(i) + '/' + filename for filename in train_list]
            self.all_train_img_list = self.all_train_img_list + train_list
            self.all_train_label_list = self.all_train_label_list + [i] * len(train_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.all_train_img_list[idx], cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        img = (img/255.).astype(np.float32)
        label = self.all_train_label_list[idx]
        label = np.eye(self.n_classes)[label].astype(np.float32)
        return img, label

    def __len__(self):
        return len(self.all_train_img_list)



num_epoch = 30
batch_size = 64
learning_rate = 1e-1


def run():
    model = MyNet()
    loss_func = MyLoss()
    accuracy=[]
    folder_path='C:/Users/418-11/Desktop/mnist/mnist/train/'
    train_dataset = Dataset(folder_path='C:/Users/418-11/Desktop/mnist/mnist/train/')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = Dataset(folder_path='C:/Users/418-11/Desktop/mnist/mnist/test/')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    for epoch in range(num_epoch):
        epoch_loss = []
        for it, data in enumerate(train_data_loader):
            x = data[0]
            y = data[1]

            y_pred = model(x)
            loss = loss_func(y_pred, y)

            if (it + 1) % 300 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (epoch + 1, num_epoch, it + 1, 600, np.array(epoch_loss).mean()))
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            loss.backward()

            for param in model.parameters():
                param.data -= learning_rate * param.grad.data
            epoch_loss.append(loss.item())
        val_epoch_loss = []

        correct = 0
        total = 0

        for data in val_data_loader:
            x = data[0]
            y = data[1]
            y_pred = model(x)

            _, label_pred = torch.max(y_pred, 1)
            _, label = torch.max(y, 1)

            total += label.size(0)
            correct += (label == label_pred).sum()

            loss = loss_func(y_pred, y)
            val_epoch_loss.append(loss.item())
    print('validation loss : ', np.array(val_epoch_loss).mean())
    print('epoch : [%d/%d] Accuracy: %d%%' % (epoch + 1, num_epoch, 100 * correct / total))

if __name__ == '__main__':
    run()



————————————————————————————————————————————



import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        #print(x.shape)
        x = x.view(-1,16*5*5)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return F.softmax(x, -1)

class MyNet1(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        ##여기에 네트워크 설계##
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        return F.softmax(x, -1)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, y_pred, y):
        return ((y - y_pred)**2).mean()


class Dataset:
    def __init__(self, folder_path):
        self.all_train_img_list = []
        train_list = os.listdir(folder_path)
        train_list = [folder_path + filename for filename in train_list]
        self.all_train_img_list = self.all_train_img_list + train_list

    def __getitem__(self, idx):
        img = cv2.imread(self.all_train_img_list[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        img = np.transpose(img, (2, 0, 1))
        img = (img/255.).astype(np.float32)
        img = torch.from_numpy(img)
        label_list = ['cat', 'dog']
        label = self.all_train_img_list[idx].split('/')[-1][:3]
        label = np.eye(len(label_list))[label_list.index(label)].astype(np.float32)
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.all_train_img_list)


num_epoch = 50
batch_size = 64
learning_rate = 1e-1


def run():
    model = MyNet()
    loss_func = MyLoss()
    accuracy = []
    train_dataset = Dataset(folder_path='C:/Users/418-11/Downloads/DogAndCat/train/')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = Dataset(folder_path='C:/Users/418-11/Downloads\DogAndCat/test/')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)


    for epoch in range(num_epoch):
            epoch_loss = []
            for it, data in enumerate(train_data_loader):
                x = data[0]
                y = data[1]
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                optimizer.zero_grad()
                if (it+1)  % 100 == 0:
                    print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %(epoch+1,num_epoch, it+1, len(train_data_loader),  loss ))
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()
                loss.backward()
                optimizer.step()
                for param in model.parameters():
                    param.data -= learning_rate * param.grad.data
                epoch_loss.append(loss.item())
            val_epoch_loss = []

            correct = 0
            total = 0
            for data in val_data_loader:
                x = data[0]
                y = data[1]
                y_pred = model(x)


                _, label_pred = torch.max(y_pred, 1)
                _, label = torch.max(y, 1)

                total += label.size(0)
                correct += (label == label_pred).sum()

                loss = loss_func(y_pred, y)
                val_epoch_loss.append(loss.item())
            print('validation loss : ', np.array(val_epoch_loss).mean())
            print('epoch : [%d/%d] Accuracy: %d%%'  % (epoch+1, num_epoch, 100 * correct / total))


if __name__ == "__main__":
    run()
