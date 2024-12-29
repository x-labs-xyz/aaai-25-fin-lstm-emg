"""
In this function, we load the pretrained model of LSTM and use a new CNN non-trained,
fine-tune them together with layer-wise gradient descend method.
"""

from lstm_model_fine import LSTM, MyDataset, DeviceDataLoader
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatreader import read_mat
from numpy import random
import time
import numpy as np
from itertools import combinations
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def arg_parser():
    parser = argparse.ArgumentParser(description='fine_tune')

    parser.add_argument('--window_size', '-ws', type=int, default=600, metavar='', help='window_size')
    parser.add_argument('--step_size', '-s', type=int, default=20, metavar='', help='step_size')
    parser.add_argument('--z_score_norm', '-z', type=bool, default=True, metavar='', help='z_score_normalization')

    parser.add_argument('--batch_size_list', '-b', type=int, default=250, metavar='', help='batch size list')
    parser.add_argument('--num_layers_list', '-nl', type=int, default=[3], metavar='', help='num_layers_list')
    parser.add_argument('--hidden_size_list', '-hs', type=int, default=[32], metavar='', help='hidden_size')
    parser.add_argument('--learning_rate_list', '-lr', type=float, default=[0.001], metavar='', help='learning rate')
    parser.add_argument('--topology_num_list', '-nt', type=int, default=[0], metavar='', help='num of topology')
    parser.add_argument('--linear_num_list', '-ln', type=int, default=[1], metavar='', help='num_layers_list')

    parser.add_argument('--optimizer_list', '-o', type=str, default=['adam'], metavar='', help='optimizer')
    parser.add_argument('--bidir', '-bd', type=bool, default=True, metavar='', help='bidirection')
    parser.add_argument('--reg', '-r', type=str, default='none', metavar='', help='regulation strategy')
    parser.add_argument('--lmd', '-lm', type=int, default=1e-5, metavar='', help='lambda')
    parser.add_argument('--train_set', '-tn', type=int, default=10, metavar='', help='num_training_sets')
    parser.add_argument('--epoch', '-e', type=int, default=10, metavar='', help='num_epochs')
    parser.add_argument('--tolerance', '-t', type=int, default=15, metavar='', help='tolerance')
    parser.add_argument('--delta', '-d', type=float, default=0.005, metavar='', help='delta')
    parser.add_argument('--S_or_M', '-sm', type=str, default='s', metavar='', help='s_or_m')
    parser.add_argument('--test_sub', '-ts', type=int, default=30, metavar='', help='tester')
    parser.add_argument('--test_channel', '-tc', type=int, default=0, metavar='', help='tester channel')

    parser.add_argument('--prediction', '-p', type=int, default=0, metavar='', help='predict next p windows')
    parser.add_argument('--percentage', '-pc', type=int, default=1, metavar='', help='6 levels, from 0-6 means from 0% to 100%')

    args = parser.parse_args()

    return args

args = arg_parser()
"""
Random seed allocation
"""
rnd = random.randint(1, 114514)
np.random.seed(rnd)
torch.manual_seed(rnd)
print(args)

def data_preprocessing(args, data_set, subject):
    """
    the input size = [num_windows, window_size, num_input_channel]
    the label size = [num_windows]
    """
    data_dir = 'data_dir'
    inputs = []
    labels = []
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
                    inputs.append(
                        df5.iloc[args.step_size * step:args.step_size * step + args.window_size, 2:].values)
                    labels.append(restimulus)
    inputs = np.array(inputs)
    labels = np.array(labels)
    if args.prediction != 0:
        labels = labels[args.prediction:]
        inputs = inputs[:-args.prediction]
    labels = labels-1
    return inputs, labels

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    # return 'cpu'


def generate_topology(num):
    topologies = []
    ls = ["mean", "var", "square"]
    if num != 0:
        topologies = combinations(ls, num)
    else:
        topologies.append([])
    return list(topologies)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


class fln1(nn.Module):
    """
    This network is clustering features as a group for each channel
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
        out = out[-args.batch_size_list[0]:]
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

"""
Designating fine-tuning repetitions
"""
if args.test_sub <= 25:
    ls = [6]
else:
    ls = [1, 3, 4, 6]

if args.percentage != 0:
    test_size = args.percentage*0.2*0.99+0.005
    X_train, y_train = data_preprocessing(args, ls, args.test_sub)
    print("X_train_size=", X_train.shape)
    _, X_train, __, y_train = train_test_split(X_train, y_train, test_size=test_size, random_state=True)
    X_test, y_test = data_preprocessing(args, (2, 5), args.test_sub)

    random_indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[random_indices]
    y_train = y_train[random_indices]
    random_indices = np.random.permutation(X_test.shape[0])
    X_test = X_test[random_indices]
    y_test = y_test[random_indices]

    print("X_train_size=", X_train.shape)
    print("X_test_size=", X_test.shape)
    num = X_train.shape[0] % args.batch_size_list
    X_train = X_train[:-num]
    y_train = y_train[:-num]
    num = X_test.shape[0] % args.batch_size_list
    X_test = X_test[:-num]
    y_test = y_test[:-num]
    print("X_train_size=", X_train.shape)
    print("X_test_size=", X_test.shape)

    if args.z_score_norm:
        for i in range(12):
            X_train[:, :, i] = preprocessing.scale(X_train[:, :, i])
            X_test[:, :, i] = preprocessing.scale(X_test[:, :, i])

    S_or_M = args.S_or_M

    tt_set = args.test_sub
    batch_size = args.batch_size_list
    num_layers = args.num_layers_list[0]
    hidden_size = args.hidden_size_list[0]
    topology_num = args.topology_num_list[0]
    topologies = generate_topology(topology_num)
    topology = topologies[0]
    linear_num = args.linear_num_list[0]
    learning_rate = args.learning_rate_list[0]
    optimizer = args.optimizer_list[0]
    ch = args.test_channel
    chn = ch + 2
    num_epochs = args.epoch
    input_size = 1
    num_classes = 12
    device = get_default_device()
    criterion = torch.nn.MSELoss()
    train_list = range(1, args.train_set + 1)

    train_dataset = MyDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    train_dl = DeviceDataLoader(train_dataloader, device)

    test_dataset = MyDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    val_dl = DeviceDataLoader(test_dataloader, device)

    """
    create models
    """
    rms_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    ssi_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    var_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    ent_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    models = [rms_models, ssi_models, var_models, ent_models]
    param = []
    for i in range(4):
        for j in range(12):
            # param.append(models[i][j].parameters())
            param += [{'params': models[i][j].parameters()}]

    if optimizer == "adam":
        optimizer1 = torch.optim.Adam(param, lr=learning_rate, weight_decay=learning_rate / 50)
    if optimizer == "SGD":
        optimizer1 = torch.optim.SGD(param, lr=learning_rate)

    channel_num = 12
    mid_nodes = 17
    output_size = 17
    topology = [2, 4, 2, 1]
    learning_rate = 0.01
    fln_model = to_device(fln1(mid_nodes, channel_num, output_size, num_features=4, topology=topology), device)
    optimizer2 = torch.optim.Adam(fln_model.parameters(), lr=0.01, weight_decay=0.01 / 50)

    """
    load state params
    """
    best_acc = -np.inf
    least_loss = np.inf
    history = []
    for j in range(12):
        models[0][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'rmsp' + str(args.prediction) + '.pth'))
        models[0][j].to(device)
    for j in range(12):
        models[1][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'ssip' + str(args.prediction) + '.pth'))
        models[1][j].to(device)
    for j in range(12):
        models[2][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'varp' + str(args.prediction) + '.pth'))
        models[2][j].to(device)
    for j in range(12):
        models[3][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'entp' + str(args.prediction) + '.pth'))
        models[3][j].to(device)

    fln_model.load_state_dict(torch.load(
        'CNN_model_dir' + 'CNN[' + str(args.test_sub) + ']pc'+str(args.percentage)+'p' + str(
            args.prediction) + '.pth'))

    """
    fine tuning
    """
    # start = time.time()
    best_acc = 0
    least_loss = np.inf
    ave_time = 0
    save_dir = "output_save_dir"
    for epoch in range(num_epochs):
        print("epoch start: " + str(epoch))
        train_acc = []
        train_losses = []
        test_acc = []
        test_losses = []
        st = time.time()
        for batch in train_dl:
            output_list = torch.empty(batch_size, 4, 12, num_classes).cuda()
            for i in range(4):
                for j in range(12):
                    x = torch.unsqueeze(batch[0][:, :, j], 2)
                    output_list[:, i, j, :] = torch.squeeze(models[i][j](x))
            output = fln_model(output_list)
            out = output[-batch_size:]
            loss = F.cross_entropy(out, batch[1])
            acc = accuracy(out, batch[1])
            train_acc.append(acc)
            train_losses.append(loss)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
        epoch_train_loss = torch.stack(train_losses).mean()
        epoch_train_acc = torch.stack(train_acc).mean()
        fin = time.time()
        ave_time = (epoch * ave_time + (fin - st)) / (epoch + 1)
        print("ave_time:", ave_time)
        print("training process complete")

        with torch.no_grad():
            for batch in val_dl:
                output_list = torch.empty(batch_size, 4, 12, num_classes).cuda()
                for i in range(4):
                    for j in range(12):
                        x = torch.unsqueeze(batch[0][:, :, j], 2)
                        output_list[:, i, j, :] = torch.squeeze(models[i][j](x))
                output = fln_model(output_list)
                out = output[-batch_size:]
                loss = F.cross_entropy(out, batch[1])
                acc = accuracy(out, batch[1])
                test_acc.append(acc)
                test_losses.append(loss)
            epoch_test_loss = torch.stack(test_losses).mean()
            epoch_test_acc = torch.stack(test_acc).mean()


        print("Epoch [{}]: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, epoch_train_acc, epoch_train_loss, epoch_test_loss, epoch_test_acc))

        history.append({"train_loss": epoch_train_loss, "train_acc": epoch_train_acc, "val_loss": epoch_test_loss,
                        "val_acc": epoch_test_acc})
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc

        if epoch_train_loss < least_loss:
            least_loss = epoch_train_loss
            for i in range(4):
                if i == 0:
                    feature_name = "RMS"
                elif i == 1:
                    feature_name = "SSI"
                elif i == 2:
                    feature_name = "VAR"
                elif i == 3:
                    feature_name = "ENT"
                for j in range(12):
                    model_name = "fine_tuned_" + feature_name + "Pc_chn_" + str(j) + '_sub' + str(
                        args.test_sub) + '_p' + str(args.prediction) + '-pc' + str(args.percentage) + ".pth"
                    torch.save(models[i][j].state_dict(), save_dir + model_name)
            torch.save(fln_model.state_dict(),
                       save_dir + str(args.test_sub) + '_p' + str(
                           args.prediction) + '-pc' + str(args.percentage) + ".pth")


if args.percentage == 0:
    X_test, y_test = data_preprocessing(args, (2, 5), args.test_sub)


    random_indices = np.random.permutation(X_test.shape[0])
    X_test = X_test[random_indices]
    y_test = y_test[random_indices]

    num = X_test.shape[0] % args.batch_size_list
    X_test = X_test[:-num]
    y_test = y_test[:-num]
    print("X_train_size=", X_train.shape)
    print("X_test_size=", X_test.shape)

    if args.z_score_norm:
        for i in range(12):
            # X_train[:, :, i] = preprocessing.scale(X_train[:, :, i])
            X_test[:, :, i] = preprocessing.scale(X_test[:, :, i])

    S_or_M = args.S_or_M

    tt_set = args.test_sub
    batch_size = args.batch_size_list
    num_layers = args.num_layers_list[0]
    hidden_size = args.hidden_size_list[0]
    topology_num = args.topology_num_list[0]
    topologies = generate_topology(topology_num)
    topology = topologies[0]
    linear_num = args.linear_num_list[0]
    learning_rate = args.learning_rate_list[0]
    optimizer = args.optimizer_list[0]
    ch = args.test_channel
    chn = ch + 2
    num_epochs = args.epoch
    input_size = 1
    num_classes = 12
    device = get_default_device()
    criterion = torch.nn.MSELoss()

    test_dataset = MyDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    val_dl = DeviceDataLoader(test_dataloader, device)

    """
    create models
    """
    rms_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    ssi_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    var_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    ent_models = [to_device(
        LSTM(num_classes, input_size, hidden_size, num_layers, 600,
             batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device) for i in range(12)]
    models = [rms_models, ssi_models, var_models, ent_models]
    param = []
    for i in range(4):
        for j in range(12):
            param += [{'params': models[i][j].parameters()}]

    if optimizer == "adam":
        optimizer1 = torch.optim.Adam(param, lr=learning_rate, weight_decay=learning_rate / 50)
    if optimizer == "SGD":
        optimizer1 = torch.optim.SGD(param, lr=learning_rate)

    channel_num = 12
    mid_nodes = 17
    output_size = 17
    topology = [2, 4, 2, 1]
    learning_rate = 0.01
    fln_model = to_device(fln1(mid_nodes, channel_num, output_size, num_features=4, topology=topology), device)
    optimizer2 = torch.optim.Adam(fln_model.parameters(), lr=0.01, weight_decay=0.01 / 50)

    """
    load state params
    """
    best_acc = -np.inf
    least_loss = np.inf
    history = []
    for j in range(12):
        models[0][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'rmsp' + str(args.prediction) + '.pth'))
        models[0][j].to(device)
    for j in range(12):
        models[1][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'ssip' + str(args.prediction) + '.pth'))
        models[1][j].to(device)
    for j in range(12):
        models[2][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'varp' + str(args.prediction) + '.pth'))
        models[2][j].to(device)
    for j in range(12):
        models[3][j].load_state_dict(
            torch.load(
                'lstm_model_dir' + 'entp' + str(args.prediction) + '.pth'))
        models[3][j].to(device)

    fln_model.load_state_dict(torch.load(
        'CNN_model_dir' + 'CNN[' + str(args.test_sub) + ']p' + str(
            args.prediction) + '.pth'))

    """
    fine tuning
    """
    # start = time.time()
    best_acc = 0
    least_loss = np.inf
    ave_time = 0
    save_dir = "save_dir"
    # for epoch in range(num_epochs):
        # start = time.time()
    print("epoch start: " + str(epoch))
    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    st = time.time()
    with torch.no_grad():
        for batch in val_dl:
            output_list = torch.empty(batch_size, 4, 12, num_classes).cuda()
            for i in range(4):
                for j in range(12):
                    x = torch.unsqueeze(batch[0][:, :, j], 2)
                    output_list[:, i, j, :] = torch.squeeze(models[i][j](x))
            # output_list = torch.tile(output_list, (12, 1, 1, 1))
            # output_list = output_list.permute(1, 2, 3, 0)
            output = fln_model(output_list)
            out = output[-batch_size:]
            loss = F.cross_entropy(out, batch[1])
            acc = accuracy(out, batch[1])
            test_acc.append(acc)
            test_losses.append(loss)
        epoch_test_loss = torch.stack(test_losses).mean()
        epoch_test_acc = torch.stack(test_acc).mean()



    print("Epoch [{}]: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, epoch_train_acc, epoch_train_loss, epoch_test_loss, epoch_test_acc))

    history.append({"train_loss": epoch_train_loss, "train_acc": epoch_train_acc, "val_loss": epoch_test_loss,
                    "val_acc": epoch_test_acc})
    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc

"""
for each percentage, each prediction, and each subject, 
there's txt recording your final test accuracy of the FIN+CNN model
"""

file_name = 'save_dir' + str(args.prediction) + '/final_result' + 'sub' + str(
    args.test_sub) + 'pc' + str(args.percentage) + '.txt'
with open(file_name, 'w') as file:
    file.write(str(best_acc.cpu().numpy()))
file.close()