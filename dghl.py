
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
    data = sio.loadmat('./Datasets/ALLASD{}_NETFC_SG_Pear.mat'.format(fold+1))
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.E2E = nn.Conv2d(1, 32, kernel_size=(90, 1))
        self.E2N = nn.Conv2d(32, 64, kernel_size=(90, 1))
        self.N2G = nn.Conv2d(64, 128, kernel_size=(90, 1))
        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 2)
        self.softmax = nn.Softmax(dim=1)
        self.hash = nn.Linear(96, 24)
        self.reg = nn.Linear(24, 8)
        self.dropout = nn.Dropout(p=0.3)

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
        x0 = self.softmax(self.fc2(x))
        x1 = self.hash(x)
        x2 = self.reg(x1)
        return x0, x1, x2


def comput_similarity(n, label):
    sim = torch.zeros([n, n]).type(torch.FloatTensor)
    for i in range(n):
        for j in range(n):
            if label[i, 0] == label[j, 0]:
                sim[i, j] = 1
            else:
                sim[i, j] = 0
    return sim


def train(train_loader, net, optimizer, loss_func, reg_loss, sim_loss):
    loss_sum = 0.
    train_hashcode = torch.empty(len(train_loader.dataset), 24)
    train_hashcode_y = torch.empty(len(train_loader.dataset))
    net.train()
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        sim = comput_similarity(batch_y.size(0), batch_y)
        if use_cuda:
            sim = sim.cuda()
        optimizer.zero_grad()
        out, out_hash, out_reg = net(batch_x)
        label = batch_y[:, 0].long()
        in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
        loss_step = 100*loss_func(out, label) + sim_loss(in_pro, sim) + 10*reg_loss(out_reg, batch_y)
        hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)
        loss_step.backward(retain_graph=True)
        optimizer.step()
        loss_sum += loss_step.data.item()
        train_hashcode[step * train_loader.batch_size:step * train_loader.batch_size + len(batch_x), :] = hashcode
        train_hashcode_y[step * train_loader.batch_size:step * train_loader.batch_size + len(batch_x)] = batch_y[:, 0]

    loss = loss_sum / len(train_loader)

    return loss, train_hashcode, train_hashcode_y


def infer(loader, net, loss_func, reg_loss, sim_loss):
    loss_sum = 0.
    valid_hashcode = torch.empty(len(loader.dataset), 24)
    valid_hashcode_y = torch.empty(len(loader.dataset))
    net.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            sim = comput_similarity(batch_y.size(0), batch_y)
            if use_cuda:
                sim = sim.cuda()
            out, out_hash, out_reg = net(batch_x)
            label = batch_y[:, 0].long()
            in_pro = torch.mm(out_hash, out_hash.transpose(1, 0))
            loss_step = 100*loss_func(out, label) + sim_loss(in_pro, sim) + 10*reg_loss(out_reg, batch_y)
            hashcode = torch.div(torch.add(torch.sign(torch.sub(torch.sigmoid(out_hash), 0.5)), 1), 2)
            loss_sum += loss_step.data.item()
            valid_hashcode[step * loader.batch_size:step * loader.batch_size + len(batch_x), :] = hashcode
            valid_hashcode_y[step * loader.batch_size:step * loader.batch_size + len(batch_x)] = batch_y[:, 0]

        loss = loss_sum / len(loader)

    return loss, valid_hashcode, valid_hashcode_y


def comput_accuracy(output, target, train_y, y):
    y_ = torch.empty(y.shape[0])
    for i in range(y.shape[0]):
        y1 = output[i, :].type(torch.int)
        hm = y1 ^ target.type(torch.int)  # ^表示异或,也就是相同为0,不同为1
        dist = hm.sum(1)
        min = torch.min(dist)
        pos = []
        for k, x in enumerate(dist):
            if x == min:
                pos.append(k)
        label = []
        for t in range(len(pos)):
            label.append(train_y[pos[t]])
        if label.count(0) > label.count(1):  # >=
            y_[i] = 0
        else:
            y_[i] = 1
    correct_prediction = y.type(torch.int) ^ y_.type(torch.int)
    acc = 1 - sum(correct_prediction) / y.shape[0]
    return acc.item()


if __name__ == '__main__':
    # 定义参数
    fold = 5
    BATCH_SIZE = 72
    EPOCH = 10000
    n_evaluation_epochs = 2
    n_patience = 50
    ax = [[] for _ in range(fold)]
    ay = [[] for _ in range(fold)]
    color_list = ['royalblue', 'darkorange', 'mediumpurple', 'forestgreen', 'firebrick']

    # 实例化网络，并且定义loss和优化器
    net = Net()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True
    loss_func = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss(reduction='mean')
    sim_loss = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=2e-4, momentum=0.9, weight_decay=1e-2)  # 1e-4


    for f in range(fold):
        net.apply(init_weights)
        print("partition {}".format(f + 1))
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2 = load_data(f)
        train_dataset = Data.TensorDataset(X_train, Y_train.to(torch.float32))
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        valid_dataset = Data.TensorDataset(X_valid, Y_valid.to(torch.float32))
        valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = Data.TensorDataset(X_test, Y_test.to(torch.float32))
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_valid_acc = 0.0
        min_valid_loss = 10000

        for epoch in range(EPOCH):
            train_loss, train_hashcode, train_hashcode_y = train(train_loader, net, optimizer, loss_func, reg_loss, sim_loss)
            print("epoch {} : train_loss= {}".format(epoch + 1, train_loss))

            valid_loss, valid_hashcode, valid_hashcode_y = infer(valid_loader, net, loss_func, reg_loss, sim_loss)
            valid_acc = comput_accuracy(valid_hashcode, train_hashcode, train_hashcode_y, valid_hashcode_y)
            print('valid_loss = ', valid_loss, 'valid_acc = ', valid_acc)

            # if epoch % n_evaluation_epochs == 0:  # 实时显示损失值曲线
            #     plt.figure(1)
            #     ax.append(epoch + 1)
            #     ay.append(np.mean(train_loss))
            #     plt.clf()
            #     plt.plot(ax, ay)
            #     plt.pause(0.01)
            #     plt.ioff()

            if epoch % n_evaluation_epochs == 0:
                ax[f].append(epoch + 1)
                ay[f].append(np.mean(train_loss))
                plt.clf()
                for i in range(fold):
                    if len(ax[i]) > 0:
                        plt.plot(ax[i], ay[i], color=color_list[i], label=f'Fold {i + 1}')
                plt.legend()
                plt.pause(0.01)
                plt.ioff()

            if valid_loss <= min_valid_loss:
                p = 0

                min_valid_loss = valid_loss
                best_valid_acc = valid_acc

            else:
                p += 1

                if p > n_patience:
                    test_loss, test_hashcode, test_hashcode_y = infer(test_loader, net, loss_func, reg_loss, sim_loss)
                    test_acc = comput_accuracy(test_hashcode, train_hashcode, train_hashcode_y, test_hashcode_y)
                    print('test_loss = ', test_loss, 'test_acc = ', test_acc)
                    break
            print("p =", p)
        print("fold {} : valid_acc= {}; test_acc= {}".format(f + 1, valid_acc, test_acc))
    # 保存图像
    plt.figure(1)
    for i in range(fold):
        if len(ax[i]) > 0:
            plt.plot(ax[i], ay[i], color=color_list[i], label=f'Fold {i + 1}')
    plt.savefig('plot.png')
    plt.ioff()
    plt.show()



