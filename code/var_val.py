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
    parser = argparse.ArgumentParser(description='var_args')

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


def VAR(x):
    """
    Variance function
    """
    VAR = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        VAR[k] = np.var(x[k, :])
    return VAR

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
    #   _, preds = torch.max(outputs, dim=1)
    # acc = torch.tensor(1-torch.abs(outputs-labels)/torch.abs(labels))
    acc = torch.tensor(1-torch.square(outputs-labels)/torch.square(labels-torch.mean(labels)))
    return acc.clamp(min=0)
    # acc = torch.tensor(1-torch.abs(outputs-labels)/torch.abs(labels))
    # if acc <= 0:

    # return torch.tensor(1-torch.abs(outputs-labels)/torch.abs(labels))


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



"""
main function
"""

save_dir = 'dir of saving your fig'
S_or_M = args.S_or_M

"""
param settings are same as feature training
"""
tt_set = range(1, 40) # you may need to change the range to select the subjects you are going to validate
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
model_name = 'var' + 'p' + str(args.prediction)
print(str(tt_set)+'\n', str(batch_size)+'\n', str(num_layers)+'\n', str(hidden_size)+'\n',
      str(topology_num)+'\n', str(topology)+'\n', str(linear_num)+'\n', str(learning_rate)+'\n',str(optimizer),str(ch))
train_list = range(1, args.train_set+1)
model = to_device(
    LSTM(num_classes, input_size, hidden_size, num_layers, 600,
         batch_size, device, topology, linear_num, args.bidir, args.reg, args.lmd), device)
model.load_state_dict(torch.load(
            'model_dir' + str(args.prediction) + '.pth'))
model.eval()
model.to(device)
print("\n" + str(model))
criterion = torch.nn.MSELoss()

"""
constructing testing dataset
"""
X, y = data_preprocessing(args, (2, 5), tt_set, batch_size=batch_size, s_or_m='m')
X_test, _, y_test, __ = train_test_split(X, y, test_size=0.1, random_state=True)
n_te = X_test.shape[0] % batch_size
y_test = y_test[:-n_te]
X_test = X_test[:-n_te]
print('X-test.shape = ', X_test.shape)
X_test = np.transpose(X_test, (0, 2, 1))
X_test = np.reshape(X_test, (-1, 600))
if args.z_score_norm:
    X_test = preprocessing.scale(X_test)
X_test_VAR = VAR(X_test)
X_test_VAR = (X_test_VAR-0.3957438647053795)/0.9240087261550427 # z-score norm with mean and sigma same as training set.
x_te = X_test
y_te = X_test_VAR
x_te = np.expand_dims(x_te, axis=2)
y_te = np.tile(y_te, (1, num_classes))
print("y_te.shape", y_te.shape)
x_te = x_te.astype("float32")
y_te = y_te.astype("float32")
test_dataset = MyDataset(x_te, y_te)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dl = DeviceDataLoader(test_dataloader, device)


"""
validation process
"""
st = time.time()
acc_list = []
for batch in val_dl:
    out = model(batch[0])
    acc = accuracy(out, batch[1])
    acc_list.append(acc)
print(len(acc_list))
print(acc_list)
# mean_acc = torch.mean(torch.stack(acc_list))
# std_acc = torch.std(torch.stack(acc_list))
total_tensor = torch.cat(acc_list, dim=0)
mean_acc = total_tensor.mean()
std_acc = total_tensor.std()
print("mean-acc=", mean_acc)
print("std-acc=", std_acc)
my_array = torch.cat(acc_list, dim=0).cpu().numpy()
np.savetxt('VAR0_R2'+str(args.prediction)+'.txt', my_array, fmt='%.4f')

"""
plot the distribution
"""

import seaborn as sns
print(my_array.shape)
sns.distplot(my_array, rug=False)
# plt.xlim(0, 1)
# plt.xticks([i/10 for i in range(11)])
plt.title("R2 VAR Distribution")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.savefig("var0_R2"+str(args.prediction)+".png")