
import scipy.io as sio
import sklearn.preprocessing as prep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.init as init


# torch.set_default_tensor_type(torch.DoubleTensor)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)



def load_data(fold):
    data = sio.loadmat('./Datasets/ALLASD{}_NETFC_SG_Pear.mat'.format(fold+1))
    X = data['net']
    X_train = data['net_train']
    X_valid = data['net_valid']
    X_test = data['net_test']

    Y_train2 = data['phenotype_train'][:, 2]
    Y_valid2 = data['phenotype_valid'][:, 2]
    Y_test2 = data['phenotype_test'][:, 2]

    ln = nn.LayerNorm(normalized_shape=[90, 90], elementwise_affine=False)
    X_train = ln(torch.tensor(X_train)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    X_valid = ln(torch.tensor(X_valid)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    X_test = ln(torch.tensor(X_test)).view(-1, 1, 90, 90).type(torch.FloatTensor)
    Y_train2 = torch.tensor(Y_train2)
    Y_valid2 = torch.tensor(Y_valid2)
    Y_test2 = torch.tensor(Y_test2)

    return X, X_train, X_valid, X_test, Y_train2, Y_valid2, Y_test2


# class E2EBlock(torch.nn.Module):
#     '''E2Eblock.'''

#     def __init__(self, in_planes, planes, bias=False):
#         super(E2EBlock, self).__init__()
#         self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, 90), bias=bias)
#         self.cnn2 = torch.nn.Conv2d(in_planes, planes, (90, 1), bias=bias)

        
#     def forward(self, x):
#         a = self.cnn1(x)
#         b = self.cnn2(x)
#         return torch.cat([a] * 90, 3) + torch.cat([b] * 90, 2)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.E2E = nn.Conv2d(1, 32, kernel_size=(90, 1))
        # self.E2E = E2EBlock(1, 32, bias=True)
        self.E2N = nn.Conv2d(32, 64, kernel_size=(90, 1))
        self.N2G = nn.Conv2d(64, 128, kernel_size=(90, 1))
        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
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


if __name__ == '__main__':
    # 定义参数
    fold = 5
    BATCH_SIZE = 96
    EPOCH = 500
    n_evaluation_epochs = 2
    n_patience = 50
    ax = [[] for _ in range(fold)]
    ay = [[] for _ in range(fold)]
    color_list = ['royalblue', 'darkorange', 'mediumpurple', 'forestgreen', 'firebrick']
    acc_list = []

    # 实例化网络，并且定义loss和优化器
    net = Net()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2)  # 1e-4


    for f in range(fold):
        net.apply(init_weights)
        X, X_train, X_valid, X_test, Y_train2, Y_valid2, Y_test2 = load_data(f)
        train_dataset = Data.TensorDataset(X_train, Y_train2.long())
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        valid_dataset = Data.TensorDataset(X_valid, Y_valid2.long())
        valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = Data.TensorDataset(X_test, Y_test2.long())
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_valid_acc = 0.0
        min_valid_loss = 10000
        p = 0
        print("-----------------------------------------------------")
        print("fold:", f+1)
        print("-----------------------------------------------------")

        for epoch in range(EPOCH):
            train_loss, train_acc = train(train_loader, net, optimizer, loss_func)
            print("epoch {} : train_loss= {}; train_acc= {}".format(epoch + 1, train_loss, train_acc))
            valid_loss, valid_acc = infer(valid_loader, net, loss_func)
            print('valid_loss = ', valid_loss, 'valid_acc = ', valid_acc, "p =", p)

            if epoch % n_evaluation_epochs == 0:
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
            if valid_loss <= min_valid_loss:
                p = 0

                min_valid_loss = valid_loss
                best_valid_acc = valid_acc

            else:
                p += 1

                if p > n_patience:
                    test_loss, test_acc = infer(test_loader, net, loss_func)
                    print('test_loss = ', test_loss, 'test_acc = ', test_acc)
                    print("fold {} : train_acc= {}; valid_acc= {}; test_acc= {}".format(f + 1, train_acc, valid_acc, test_acc))
                    acc_list.append(test_acc)
                    break
    
    print("ave_test_acc =", round(sum(acc_list) / len(acc_list), 5))
    # 保存图像
    plt.savefig('plot.png')
    plt.ioff()
    plt.show()


"""
EPOCH = 500:
    lr = 1e-3, wd = 1e-2, drop_out = 0.4: ave_test_acc = 63.803%
    lr = 1e-3, wd = 1e-2, drop_out = 0.5: ave_test_acc = 64.679%
    lr = 1e-3, wd = 1e-2, drop_out = 0.6: ave_test_acc = 64.00%
"""
