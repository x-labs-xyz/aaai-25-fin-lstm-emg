"""
This is the function of shadow training the downstream CNN, this part is distached with the LSTM
"""

import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from pymatreader import read_mat
import os
import numpy as np
from numpy import random
from scipy.special import softmax
from scipy.stats import entropy
import time
from scipy.stats import skew
from scipy.stats import kurtosis
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='fine_tune')
    parser.add_argument('--percentage', '-pc', type=int, default=0, metavar='',
                        help='6 levels, from 0-5 means from 0% to 100%')
    parser.add_argument('--prediction', '-p', type=int, default=0, metavar='', help='predict next p windows')
    args = parser.parse_args()
    return args


arg = arg_parser()
args.prediction = arg.prediction


# import scipy.special.softmax as softmax
def RMS(x):
    RMS = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            # x[k,:,j] = softmax(x[k,:,j])
            RMS[k, j] = np.sqrt(np.mean(np.square(x[k, :, j])))
    return np.expand_dims(RMS, axis=1)


def VAR(x):
    VAR = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            # x[k,:,j] = softmax(x[k,:,j])
            VAR[k, j] = np.var(x[k, :, j])
    return np.expand_dims(VAR, axis=1)


def MAV(x):
    MAV = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            MAV[k, j] = np.mean(np.abs(x[k, :, j]))
    return np.expand_dims(MAV, axis=1)


def SSI(x):
    SSI = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            # x[k,:,j] = softmax(x[k,:,j])
            SSI[k, j] = np.sum(np.square(x[k, :, j]))
    return np.expand_dims(SSI, axis=1)


def KUR(x):
    KUR = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            KUR[k, j] = kurtosis(x[k, :, j])
    return np.expand_dims(KUR, axis=1)


def ENT(x):
    ENT = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            sm = softmax(x[k, :, j])
            ENT[k, j] = entropy(sm)
    return np.expand_dims(ENT, axis=1)


def WL(x):
    WL = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            a = 0
            for i in range(x.shape[1] - 1):
                a = a + abs(x[k, i + 1, j] - x[k, i, j])
            WL[k, j] = a
    return np.expand_dims(WL, axis=1)


def SKEW(x):
    SKEW = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            SKEW[k, j] = skew(x[k, :, j])
    return np.expand_dims(SKEW, axis=1)


def ZCR(x):
    ZCR = np.zeros([x.shape[0], x.shape[2]])
    for k in range(x.shape[0]):
        for j in range(x.shape[2]):
            zcr = 0
            for i in range(1, x.shape[1]):
                if x[k, i - 1, j] * x[k, i, j] < 0:
                    zcr += 1
            ZCR[k, j] = zcr / 600
    return np.expand_dims(ZCR, axis=1)


class args:
    window_size = 600
    step_size = 20
    batch_size = 300
    dilation = 0
    num_layers = 3
    z_score_norm = True
    hidden_size = 128
    input_size = 12
    num_kernels = (32, 64, 64, 128, 128, 256, 256)
    list_dilation = (1, 2, 4, 8, 8, 8, 1)

    learning_rate = 0.01
    factor = 0.5
    patience = 50
    threshold = 1e-2
    lr_limit = 1e-4
    measurement = 'min'
    scheduler = 'plateau'
    optimizer = 'adam'
    progress_bar = False
    cuda = True
    test = False
    plot = False
    save = False
    scheduler = 'plateau'
    weight_decay = 3e-5
    prediction = 0

    training_set = (1, 3)
    subject_list = [1]
    testing_set = (2, 5)
    epoch = 15
    cuda = cuda and torch.cuda.is_available()
    weight_decay = float(weight_decay)

    np.random.seed(114514)


def data_preprocessing(args, data_set, subject_list):
    data_dir = 'your database dir'
    inputs = []
    labels = []

    for subject in subject_list:
        f = data_dir + '/S' + str(subject) + '_E1_A1.mat'
        data_raw = read_mat(f)
        emg = data_raw['emg']
        df1 = pd.DataFrame(emg)
        df2 = pd.DataFrame(data_raw['restimulus'])
        df3 = pd.DataFrame(data_raw['repetition'])
        df = pd.concat([df3, df2, df1], axis=1)
        for repetition in data_set:
            for restimulus in range(1, 18):
                df4 = df.loc[df.iloc[:, 0] == repetition, :]
                df5 = df4.loc[df.iloc[:, 1] == restimulus, :]
                for step in range((df5.shape[0] - args.window_size) // args.step_size):
                    if df5.iloc[args.step_size * step, 1] == df5.iloc[args.step_size * step + args.window_size, 1] and \
                            df5.iloc[args.step_size * step, 0] == df5.iloc[args.step_size * step + args.window_size, 0]:
                        a = df5.iloc[args.step_size * step:args.step_size * step + args.window_size, 2:].values
                        inputs.append(a)
                        labels.append(restimulus)
    inputs = np.array(inputs)
    labels = np.array(labels)
    if args.prediction != 0:
        labels = labels[args.prediction:]
        inputs = inputs[:-args.prediction]
    labels = labels - 1

    return inputs, labels


from torch.utils.data import Dataset, TensorDataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, targets):  # , transform=None
        self.data = data
        self.labels = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.from_numpy(np.array(y))
        return x, y

    def __len__(self):
        return len(self.data)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    # return 'cpu'


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


from torch.autograd import Variable


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


device = get_default_device()
device


class fln1(nn.Module):
    """
    Feature Learning Network,
    transforming features to movements classification.
    """

    def __init__(self, mid_nodes, channel_num, output_size, num_features, topology):
        super(fln1, self).__init__()
        self.mid_nodes = mid_nodes
        self.channel_num = channel_num
        self.output_size = output_size
        self.num_features = num_features
        self.topology = topology

        self.conv1 = nn.Conv2d(4, 64, 5)  # 12*12->8*8
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 32, 5)  # 8*8->4*4
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 17, 4)  # 4->1
        self.batchnorm3 = nn.BatchNorm2d(17)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        out = self.prelu3(x)
        out = torch.squeeze(out)

        return out

    def training_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        out = out[-args.batch_size:]
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)

        return acc, loss

    def validation_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}]: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_acc'], result['train_loss'], result['val_loss'], result['val_acc']))




import openpyxl

workbook = openpyxl.Workbook()
worksheet = workbook.active

for ts in range(1, 41):
    test_sub = [ts]
    if ts <= 25:
        ls = [6]
    else:
        ls = [1, 3, 4, 6]

    test_size = arg.percentage * 0.2 * 0.99 + 0.005
    print(ts)
    print(test_size)
    X_train, y_train = data_preprocessing(args, ls, test_sub, args.batch_size, 'm')
    print("X_train_size=", X_train.shape)
    _, X_train, __, y_train = train_test_split(X_train, y_train, test_size=test_size, random_state=True)
    print("X_train_size=", X_train.shape)
    X_test, y_test = data_preprocessing(args, (2, 5), test_sub, args.batch_size, 'm')
    # X_test, _, y_test, __ = train_test_split(X, y, test_size=0.2, random_state=True)

    random_indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[random_indices]
    y_train = y_train[random_indices]
    random_indices = np.random.permutation(X_test.shape[0])
    X_test = X_test[random_indices]
    y_test = y_test[random_indices]

    num1 = X_train.shape[0] % args.batch_size
    num2 = X_test.shape[0] % args.batch_size
    X_train = X_train[:-num1]
    y_train = y_train[:-num1]

    y_test = y_test[:-num2]
    X_test = X_test[:-num2]
    # X_train = X_train[:, :, 2:]
    # X_test = X_test[:, :, 2:]

    X_train_new = np.zeros(X_train.shape)
    X_test_new = np.zeros(X_test.shape)
    for i in range(X_train.shape[2]):
        X_train_new[:, :, i] = preprocessing.scale(X_train[:, :, i])
    for i in range(X_test.shape[2]):
        X_test_new[:, :, i] = preprocessing.scale(X_test[:, :, i])
    X_train = X_train_new
    X_test = X_test_new
    X_train_RMS = RMS(X_train)
    X_train_RMS = (X_train_RMS - 0.3957438647053795) / 0.9240087261550427
    X_train_VAR = VAR(X_train)
    X_train_VAR = (X_train_VAR - 1.006967578266923) / 13.98630613347952
    # X_train_MAV = MAV(X_train)
    X_train_SSI = SSI(X_train)
    X_train_SSI = (X_train_SSI - 606.243199458427) / 8412.157798605922
    # X_train_KUR = KUR(X_train)
    X_train_ENT = ENT(X_train)
    X_train_ENT = (X_train_ENT - 6.166928238406377) / 0.7546575182578825
    X_train_features = np.concatenate([X_train_RMS, X_train_VAR, X_train_SSI, X_train_ENT], axis=1)
    X_test_RMS = RMS(X_test)
    X_test_RMS = (X_test_RMS - 0.3957438647053795) / 0.9240087261550427
    X_test_VAR = VAR(X_test)
    X_test_VAR = (X_test_VAR - 1.006967578266923) / 13.98630613347952
    # X_test_MAV = MAV(X_test)
    X_test_SSI = SSI(X_test)
    X_test_SSI = (X_test_SSI - 606.243199458427) / 8412.157798605922
    # X_test_KUR = KUR(X_test)
    X_test_ENT = ENT(X_test)
    X_test_ENT = (X_test_ENT - 6.166928238406377) / 0.7546575182578825
    X_test_features = np.concatenate([X_test_RMS, X_test_VAR, X_test_SSI, X_test_ENT], axis=1)

    x = X_train_features
    x_te = X_test_features

    y = y_train
    y_te = y_test
    x = np.tile(x, (12, 1, 1, 1))
    x_te = np.tile(x_te, (12, 1, 1, 1))
    x = np.transpose(x, (1, 2, 3, 0))
    x_te = np.transpose(x_te, (1, 2, 3, 0))
    std = x[:, :, 0, :] * 0.01
    std = np.expand_dims(std, 2)
    for i in range(1, 12):
        noise = np.random.normal(loc=0, scale=np.abs(std), size=(x.shape[0], x.shape[1], 1, x.shape[3]))
        x = x + noise
    # noise_matrix = np.random.normal(loc=0, scale=np.sqrt(0.01), size=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    # x = x + noise_matrix

    y = y.astype("int")
    x_te = x_te.astype("float32")
    y_te = y_te.astype("int")
    print(x.shape)
    print(y.shape)

    train_dataset = MyDataset(x, y)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = MyDataset(x_te, y_te)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    train_dl = DeviceDataLoader(train_dataloader, device)
    val_dl = DeviceDataLoader(test_dataloader, device)
    num_epochs = 30
    learning_rate = 0.01

    # input_size = x.shape[1]
    channel_num = 12
    mid_nodes = 17
    output_size = 17
    topology = [2, 4, 2, 1]
    num_features = 4

    model = to_device(fln1(mid_nodes, channel_num, output_size, num_features, topology), device)
    model

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    best_acc = -np.inf
    history = []
    # Train the model
    ave_time = 0
    for epoch in range(num_epochs):
        start = time.time()
        train_acc = []
        train_losses = []
        i = 0
        #   print(train_dl.__len__())
        for batch in train_dl:
            acc, loss = model.training_step(batch)
            train_acc.append(acc)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase

        result = evaluate(model, val_dl)
        result['train_acc'] = torch.stack(train_acc).mean().item()
        result['train_loss'] = torch.stack(train_losses).mean().item()

        save_dir = 'save_dir'

        if result['val_acc'] > best_acc:
            best_acc = result['val_acc']
            torch.save(model.state_dict(), save_dir + 'CNN' + str(test_sub) + 'pc' + str(arg.percentage) + 'p' + str(
                args.prediction) + '.pth')

        model.epoch_end(epoch, result)
        history.append(result)
        end = time.time()
        ave_time = (ave_time * epoch + (end - start)) / (epoch + 1)
        print("ave_time:", ave_time)

    cell = worksheet.cell(row=arg.percentage + 2, column=ts + 1)
    cell.value = best_acc
    workbook.save('CNN_per' + str(arg.percentage) + '_pre' + str(args.prediction) + '.xlsx')
