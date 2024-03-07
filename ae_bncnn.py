import scipy.io as sio
import sklearn.preprocessing as prep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fy
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.utils.data import DataLoader,TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import os


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def mapStd(X,X_train,X_valid,X_test):
    preprocessor=prep.StandardScaler().fit(X)
    X = preprocessor.transform(X)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X,X_train,X_valid,X_test


def mapMinmax(X,X_train,X_valid,X_test):
    preprocessor=prep.MinMaxScaler().fit(X)
    X = 2*preprocessor.transform(X)-1
    X_train = 2*preprocessor.transform(X_train)-1
    X_valid = 2*preprocessor.transform(X_valid)-1
    X_test = 2*preprocessor.transform(X_test)-1
    return X,X_train,X_valid,X_test


def load_data(fold):
    data = sio.loadmat('./BNC-DGHL/Datasets/ALLASD{}_NETFC_SG_Pear.mat'.format(fold+1))
    X = data['net']
    X_train = data['net_train']
    X_valid = data['net_valid']
    X_test = data['net_test']

    Idx = [2, 3, 4, 5, 6, 7, 8, 9]  # 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
    Y = data['phenotype'][:, Idx]
    Y_train = data['phenotype_train'][:, Idx]
    Y_valid = data['phenotype_valid'][:, Idx]
    Y_test = data['phenotype_test'][:, Idx]
    col_idx = [1, 4, 5, 6]  # 3:Age 6:FIQ 7:VIQ 8:PIQ
    Y[:, col_idx], Y_train[:, col_idx], \
    Y_valid[:, col_idx], Y_test[:, col_idx] = mapStd(Y[:, col_idx],
                                                     Y_train[:, col_idx],
                                                     Y_valid[:, col_idx],
                                                     Y_test[:, col_idx])
    col_idx = [2, 3, 7]
    Y[:, col_idx], Y_train[:, col_idx], \
    Y_valid[:, col_idx], Y_test[:, col_idx] = mapMinmax(Y[:, col_idx],
                                                        Y_train[:, col_idx],
                                                        Y_valid[:, col_idx],
                                                        Y_test[:, col_idx])


    Y_train2 = data['phenotype_train'][:, 2]
    Y_valid2 = data['phenotype_valid'][:, 2]
    Y_test2 = data['phenotype_test'][:, 2]

    ln = nn.LayerNorm(normalized_shape=[90, 90], elementwise_affine=False)
    X_train = ln(torch.tensor(X_train)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    X_valid = ln(torch.tensor(X_valid)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    X_test = ln(torch.tensor(X_test)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    Y_train = torch.tensor(Y_train)
    Y_valid = torch.tensor(Y_valid)
    Y_test = torch.tensor(Y_test)
    Y_train2 = torch.tensor(Y_train2)
    Y_valid2 = torch.tensor(Y_valid2)
    Y_test2 = torch.tensor(Y_test2)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.autoencoder = Autoencoder()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.E2E = nn.Conv2d(1, 32, kernel_size=(90, 1))
        # self.E2E = E2EBlock(1, 32, bias=True)
        self.E2N = nn.Conv2d(32, 64, kernel_size=(90, 1))
        self.N2G = nn.Conv2d(64, 128, kernel_size=(90, 1))
        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.autoencoder(x)
        x = F.relu(self.E2E(x) + self.E2E(x).transpose(3, 2))
        x = self.dropout(x)
        x = F.relu(self.E2N(x).transpose(3, 2)*2)
        x = self.dropout(x)
        x = F.relu(self.N2G(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x


def train(train_loader, net, optimizer, loss_func):
    acc_sum = 0.
    loss_sum = 0.
    net.train()
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        optimizer.zero_grad()
        out = net(batch_x)
        loss_step = loss_func(out, batch_y)
        loss_step.backward()
        optimizer.step()
        acc_step = (out.argmax(dim=1) == batch_y).float().mean().item()
        loss_sum += loss_step.data.item()
        acc_sum += acc_step
        # print("step {}: loss: {}; acc: {}".format(epoch + 1, step + 1, loss_step.data.item(), acc_step))
    loss = loss_sum / len(train_loader)
    acc = acc_sum / len(train_loader)
    return loss, acc


def infer(loader, net, loss_func):
    acc_sum = 0.
    loss_sum = 0.
    net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            out = net(batch_x)
            loss_step = loss_func(out, batch_y)
            acc_step = (out.argmax(dim=1) == batch_y).float().mean().item()
            loss_sum += loss_step.data.item()
            acc_sum += acc_step
            # print("step {}: loss: {}; acc: {}".format(epoch + 1, step + 1, loss_step.data.item(), acc_step))
        loss = loss_sum / len(loader)
        acc = acc_sum / len(loader)
    return loss, acc


max_acc = 0
for i in range(1):
    # 定义参数
    fold = 5
    BATCH_SIZE = 32
    EPOCH = 2000
    n_evaluation_epochs = 2
    n_patience = 50
    ax = [[] for _ in range(fold)]
    ay = [[] for _ in range(fold)]
    color_list = ['royalblue', 'darkorange', 'mediumpurple', 'forestgreen', 'firebrick']
    acc_list = []

    # 实例化网络，并且定义loss和优化器
    net = Net().to(device)
    


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=3e-4, momentum=0.9, weight_decay=1e-2)  # 1e-4


    for f in range(fold):
        net.apply(init_weights)
        net.autoencoder.load_state_dict(torch.load("C:/Users/xhx20/Desktop/AE_DGHL.pth"))

        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2 = load_data(f)

        """只用1/5个样本进行学习"""
        fraction = 5
        X_train, X_valid, X_test = X_train[:len(X_train)//fraction], X_valid[:len(X_valid)//fraction], X_test[:len(X_test)//fraction]
        Y_train2, Y_valid2, Y_test2 = Y_train2[:len(Y_train2)//fraction], Y_valid2[:len(Y_valid2)//fraction], Y_test2[:len(Y_test2)//fraction]


        train_dataset = Data.TensorDataset(X_train, Y_train2.long())
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        valid_dataset = Data.TensorDataset(X_valid, Y_valid2.long())
        valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = Data.TensorDataset(X_test, Y_test2.long())
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_valid_acc = 0.0
        min_valid_loss = 10000
        p = 0
        # print("-----------------------------------------------------")
        # print("fold:", f+1)
        # print("-----------------------------------------------------")

        for epoch in range(EPOCH):
            train_loss, train_acc = train(train_loader, net, optimizer, loss_func)
            # print("epoch {} : train_loss= {}; train_acc= {}".format(epoch + 1, train_loss, train_acc))
            valid_loss, valid_acc = infer(valid_loader, net, loss_func)
            # print('valid_loss = ', valid_loss, 'valid_acc = ', valid_acc, "p =", p)

            if epoch % n_evaluation_epochs == 0:
                print('Epoch:', epoch, 'p:', p, 'train_loss:', train_loss, 'train_acc:', train_acc, 'valid_loss:', valid_loss, 'valid_acc:', valid_acc)
                plt.figure(1)
                ax[f].append(epoch + 1)
                ay[f].append(np.mean(train_loss))
                plt.clf()
                for i in range(fold):
                    if len(ax[i]) > 0:
                        plt.plot(ax[i], ay[i], color=color_list[i], label=f'Fold {i + 1}')
                plt.legend()
                plt.pause(0.01)
                # plt.ioff()
            if train_loss >= 0.35:
                p = 0

                # min_valid_loss = valid_loss
                # best_valid_acc = valid_acc

            else:
                p += 1

                if p > n_patience:
                    test_loss, test_acc = infer(test_loader, net, loss_func)
                    print('test_loss = ', test_loss, 'test_acc = ', test_acc)
                    # print("fold {} : train_acc= {}; valid_acc= {}; test_acc= {}".format(f + 1, train_acc, valid_acc, test_acc))
                    acc_list.append(test_acc)
                    break
        # test_loss, test_acc = infer(test_loader, net, loss_func)
        # print('test_loss = ', test_loss, 'test_acc = ', test_acc)
        # # print("fold {} : train_acc= {}; valid_acc= {}; test_acc= {}".format(f + 1, train_acc, valid_acc, test_acc))
        # acc_list.append(test_acc)

    ave_test_acc = round(sum(acc_list) / len(acc_list), 5)
    print(acc_list)
    print("ave_test_acc =", ave_test_acc)
    if ave_test_acc > max_acc:
        max_acc = ave_test_acc
        print("max:", max_acc)
        torch.save(net, 'brainnetcnn.pth')