"""
This function is the main function of learning Root Mean Square feature with FIN
"""

from lstm_model_fine import LSTM, MyDataset, DeviceDataLoader
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatreader import read_mat
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from scipy.signal import cwt, ricker
from sklearn.preprocessing import StandardScaler
import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.special import softmax
from scipy.stats import entropy
from itertools import combinations
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def arg_parser():
    parser = argparse.ArgumentParser(description='rms_args')

    parser.add_argument('--window_size', '-ws', type=int, default=600, metavar='', help='window_size')
    parser.add_argument('--step_size', '-s', type=int, default=20, metavar='', help='step_size')
    parser.add_argument('--z_score_norm', '-z', type=bool, default=True, metavar='', help='z_score_normalization')

    parser.add_argument('--batch_size_list', '-b', type=int, default=4000, metavar='', help='batch size list')
    parser.add_argument('--num_layers_list', '-nl', type=int, default=[3], metavar='', help='num_layers_list')
    parser.add_argument('--hidden_size_list', '-hs', type=int, default=[32], metavar='', help='hidden_size')
    parser.add_argument('--learning_rate_list', '-lr', type=float, default=[0.001], metavar='', help='learning rate')
    parser.add_argument('--topology_num_list', '-nt', type=int, default=[0], metavar='', help='num of topology')
    parser.add_argument('--linear_num_list', '-ln', type=int, default=[1], metavar='', help='num_layers_list')

    parser.add_argument('--optimizer_list', '-o', type=str, default=['adam'], metavar='', help='optimizer')
    parser.add_argument('--bidir', '-bd', type=bool, default=True, metavar='', help='bidirection')
    parser.add_argument('--reg', '-r', type=str, default='none', metavar='', help='regulation strategy')
    parser.add_argument('--lmd', '-lm', type=int, default=1e-5, metavar='', help='lambda')
    parser.add_argument('--train_set', '-tn', type=int, default=25, metavar='', help='num_training_sets')
    parser.add_argument('--epoch', '-e', type=int, default=60, metavar='', help='num_epochs')
    parser.add_argument('--tolerance', '-t', type=int, default=15, metavar='', help='tolerance')
    parser.add_argument('--delta', '-d', type=float, default=0.005, metavar='', help='delta')
    parser.add_argument('--S_or_M', '-sm', type=str, default='s', metavar='', help='s_or_m')
    parser.add_argument('--test_sub', '-ts', type=int, default=30, metavar='', help='tester')
    parser.add_argument('--test_channel', '-tc', type=int, default=0, metavar='', help='tester channel')

    parser.add_argument('--prediction', '-p', type=int, default=0, metavar='', help='predict next p windows')

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


def data_preprocessing(args, data_set, subject_list):
    """
    read in sEMG data
    inputs:
        data_set: designating repetitions
        subject_list: a list of subjects included in the experiment, it must be a list
    ouputs:
        inputs: (N, 600, 12) size of sEMG signal, N is the total window number, 600 is the window size, 12 is the channel number
        labels: (N,) movement lables

    """
    data_dir = 'your database dir'
    inputs = []
    labels = []

    for subject in subject_list:
        f = data_dir + '/S' + str(subject) + '_E1_A1.mat'
        data_raw = read_mat(f)
        emg = data_raw['emg']
        # if args.z_score_norm:
        #     emg = preprocessing.scale(data_raw['emg'])
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
        del data_raw
        del emg
        del a
        del df1
        del df2
        del df3
        del df
    inputs = np.array(inputs)
    labels = np.array(labels)
    if args.prediction != 0:
        labels = labels[args.prediction:]
        inputs = inputs[:-args.prediction]
    num = inputs.shape[0] % args.batch_size
    inputs = inputs[:inputs.shape[0] - num, :]
    labels = labels[:labels.shape[0] - num] - 1

    return inputs, labels


def RMS(x):
    RMS = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        RMS[k] = np.sqrt(np.mean(np.square(x[k, :])))
    return RMS

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


def accuracy(outputs, labels):
    """
    R2 accuracy
    """
    #   _, preds = torch.max(outputs, dim=1)
    return torch.tensor(1-torch.sum(torch.square(outputs-labels))/torch.sum(torch.square(labels-torch.mean(labels))))


def evaluate(model, val_loader):
    with torch.no_grad():
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def generate_topology(num):
    topologies = []
    ls = ["mean", "var", "square"]
    if num != 0:
        topologies = combinations(ls, num)
    else:
        topologies.append([])
    return list(topologies)


class EarlyStopping:
    """
    Early stopping subject
    """

    def __init__(self, tolerance, min_delta):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, history):
        if len(history) >= 2:
            if abs(history[-1].get('train_loss') - history[-2].get('train_loss')) <= self.min_delta and history[-1].get('train_acc') > 0.5:
                self.counter = self.counter + 1
                if self.counter >= self.tolerance:
                    self.early_stop = True

    def reset(self, tolerance, min_delta):
        self.__init__(tolerance, min_delta)


"""
main function
"""
save_dir = 'your direction of saving your model'
early_stopping = EarlyStopping(tolerance=args.tolerance, min_delta=args.delta)
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
chn = ch+2
num_epochs = args.epoch
input_size = 1
num_classes = 12
device = get_default_device()
model_name = 'rms' + 'p' + str(args.prediction)
print(str(tt_set)+'\n', str(batch_size)+'\n', str(num_layers)+'\n', str(hidden_size)+'\n',
      str(topology_num)+'\n', str(topology)+'\n', str(linear_num)+'\n', str(learning_rate)+'\n',str(optimizer),str(ch))
train_list = range(1, args.train_set+1)
model = to_device(
    LSTM(num_classes, input_size, hidden_size, num_layers, 600,
         batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device)
print("\n" + str(model))
criterion = torch.nn.MSELoss()
if optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate / 50)
if optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
best_acc = -np.inf
least_loss = np.inf
history = []
early_stopping.early_stop = False
# Train the model
st = time.time()
count_chn = 0   # count_chn +=1 for each epoch

"""
constructing testing dataset
"""
X, y = data_preprocessing(args, (2, 5), [tt_set], batch_size=batch_size, s_or_m='m')
X_test, _, y_test, __ = train_test_split(X, y, test_size=0.1, random_state=True)
n_te = X_test.shape[0] - (X_test.shape[0] % batch_size)
y_test = y_test[:n_te]
X_test = X_test[:n_te]
for i in range(17):
    print(
        "testing set used data from stimulation " + str(i + 1) + ':' + str(
            np.count_nonzero(y_test[:] == i)))
print('X-test.shape = ', X_test.shape)
X_test = np.transpose(X_test, (0, 2, 1))
X_test = np.reshape(X_test, (-1, 600))
if args.z_score_norm:
    X_test = preprocessing.scale(X_test)
X_test_RMS = RMS(X_test)
# if args.z_score_norm:
#     X_test = preprocessing.scale(X_test)
X_test_RMS = preprocessing.scale(X_test_RMS)
x_te = X_test
y_te = X_test_RMS
x_te = np.expand_dims(x_te, axis=2)
y_te = np.tile(y_te, (1, num_classes))
print("y_te.shape", y_te.shape)
x_te = x_te.astype("float32")
y_te = y_te.astype("float32")
test_dataset = MyDataset(x_te, y_te)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dl = DeviceDataLoader(test_dataloader, device)

"""
    constructing cycling training set
    """
X, y = data_preprocessing(args, (1, 3, 4), train_list, batch_size=batch_size, s_or_m='m')
X_train, _, y_train, __ = train_test_split(X, y, test_size=0.1, random_state=True)
n_tr = X_train.shape[0] - (X_train.shape[0] % batch_size)
X_train = X_train[:n_tr]
y_train = y_train[:n_tr]
X_train = np.transpose(X_train, (0, 2, 1))
X_train = np.reshape(X_train, (-1, 600))
print("\nTraining Shape:" + str(X_train.shape) + str(y_train.shape))
if args.z_score_norm:
    X_train = preprocessing.scale(X_train)
X_train_RMS = RMS(X_train)
# if args.z_score_norm:
#     X_train = preprocessing.scale(X_train)
X_train_RMS = preprocessing.scale(X_train_RMS)
x = X_train
y = X_train_RMS
y = np.tile(y, (1, num_classes))
x = np.expand_dims(x, axis=2)
std = y[:, 0] * 0.01
std = np.expand_dims(std, 1)
noise = np.random.normal(loc=0, scale=np.abs(std), size=(y.shape[0], y.shape[1]))
y = y + noise
x = x.astype("float32")
y = y.astype("float32")
train_dataset = MyDataset(x, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
print("\nx.shape=" + str(x.shape))
print("\ny.shape=" + str(y.shape))
train_dl = DeviceDataLoader(train_dataloader, device)

for epoch in range(num_epochs):
    st = time.time()
    print("epoch start")
    train_acc = []
    train_losses = []
    if count_chn == 12:
        count_chn = 0

    """
    training process:
    """
    for batch in train_dl:
        acc, loss = model.training_step(batch)
        train_acc.append(acc)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    count_chn = count_chn + 1
    result = evaluate(model, val_dl)
    result['train_acc'] = torch.stack(train_acc).mean().item()
    #   result['train_acc'] = 0
    result['train_loss'] = torch.stack(train_losses).mean().item()

    if result['train_loss'] < least_loss:
        least_loss = result['train_loss']
        torch.save(model.state_dict(), save_dir + model_name + '.pth')
    if result['val_acc'] > best_acc:
        best_acc = result['val_acc']
        val_loss = result['val_loss']

    print(
        "\n" + "Epoch [{}]: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_acc'], result['train_loss'],
            result['val_loss'], result['val_acc']))
    history.append(result)

    early_stopping(history)
    if early_stopping.early_stop:
        print("\n" + "We are at epoch:" + str(epoch))
        break
    print("epoch end")
    ed = time.time()
    print("time_elapsed_epoch:", ed - st)

del model
early_stopping.reset(tolerance=args.tolerance, min_delta=args.delta)

"""
Visulization
"""

train_losses = [x.get('train_loss') for x in history]
val_losses = [x['val_loss'] for x in history]
plt.plot(train_losses, '-bx')
plt.plot(val_losses, '-rx')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Training', 'Test'])
plt.title('Loss vs. No. of epochs')
plt.savefig(save_dir + model_name + 'LOSS.png')
plt.clf()

train_acc = [x.get('train_acc') for x in history]
val_acc = [x['val_acc'] for x in history]
plt.plot(train_acc, '-bx')
plt.plot(val_acc, '-rx')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Training', 'Test'])
plt.title('Accuracy vs. No. of epochs')
plt.savefig(save_dir + model_name + 'ACC.png')
plt.clf()
print("\n" + 'epoch period:' + str((ed - st) / (epoch+1)))
print("\n" + "best_acc=" + str(best_acc))
print("\n" + "least_loss=" + str(least_loss))
print("\n" + "least_val_loss=" + str(val_loss))


