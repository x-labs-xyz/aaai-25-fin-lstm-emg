import torch.nn as nn

import torch

import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    Database construction subject
    """
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


device = get_default_device()


def accuracy(outputs, labels):
    """
    Feature Imitation accuracy function,
    Input:
        outputs: FIN's prediction of a batch
        labels: Ground-truth features of a batch
    Output:
        first output: R-square accuracy of a batch
        second output: Mean Absolute Percentage accuracy of a batch
        Switch to either accuracy manually based on your desire.
    """
    #   _, preds = torch.max(outputs, dim=1)
    return torch.tensor(1-torch.sum(torch.square(outputs-labels))/torch.sum(torch.square(labels-torch.mean(labels))))
    # return torch.tensor(1-torch.abs(outputs-labels)/torch.abs(labels))


def evaluate(model, val_loader):
    with torch.no_grad():
        outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


class LSTM(nn.Module):
    """
    This is the FIN network
    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, batch_size, device, topology,
                 num_linear_layers, bidir, reg, lmd):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.topology = topology
        self.num_linear_layers = num_linear_layers
        self.bidir = bidir
        self.reg = reg
        self.lmd = lmd
        if self.bidir:
            self.D = 2
        else:
            self.D = 1

        self.Flatten = nn.Flatten(1, 2)
        self.loss_fcn = nn.MSELoss()
        self.fc = nn.Linear(hidden_size, num_classes)

        if len(topology) == 1:
            self.lstm = nn.LSTM(input_size=input_size + 1, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=self.bidir)
        if len(topology) == 2:
            self.lstm = nn.LSTM(input_size=input_size + 2, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=self.bidir)
        if len(topology) == 3:
            self.lstm = nn.LSTM(input_size=input_size + 3, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=self.bidir)
        if len(topology) == 0:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=self.bidir)

        if num_linear_layers == 1:
            self.fc1 = nn.Linear(hidden_size, num_classes)
        if num_linear_layers == 2:
            self.fc1 = nn.Linear(hidden_size, num_classes)
            self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        if num_linear_layers == 3:
            self.fc1 = nn.Linear(hidden_size, num_classes)
            self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
            self.fc3 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        if num_linear_layers == 4:
            self.fc1 = nn.Linear(hidden_size, num_classes)
            self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
            self.fc3 = nn.Linear(4 * hidden_size, 2 * hidden_size)
            self.fc4 = nn.Linear(8 * hidden_size, 4 * hidden_size)
        if num_linear_layers == 0:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_classes,
                                num_layers=num_layers, batch_first=True, bidirectional=self.bidir, proj_size=1)

    def forward(self, x):
        mean = torch.mean(x, 1)
        mean = torch.broadcast_to(mean, (mean.size(dim=0), 600))
        mean = torch.unsqueeze(mean, 2)
        var = torch.var(x, 1)
        var = torch.broadcast_to(var, (var.size(dim=0), 600))
        var = torch.unsqueeze(var, 2)
        sqr = torch.square(x)
        sqr = torch.unsqueeze(sqr, 2)


        h_0 = to_device(torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size), self.device)
        c_0 = to_device(torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size), self.device)

        if len(self.topology) != 0:
            if len(self.topology) == 1:
                if self.topology[0] == "mean":
                    cat = torch.cat([x, mean], 2)
                if self.topology[0] == "var":
                    cat = torch.cat([x, var], 2)
                if self.topology[0] == "var":
                    cat = torch.cat([x, sqr], 2)
            if len(self.topology) == 2:
                if self.topology[0] == "mean":
                    cat = torch.cat([x, mean], 2)
                if self.topology[0] == "var":
                    cat = torch.cat([x, var], 2)
                if self.topology[0] == "var":
                    cat = torch.cat([x, sqr], 2)
                if self.topology[1] == "mean":
                    cat = torch.cat([cat, mean], 2)
                if self.topology[1] == "var":
                    cat = torch.cat([cat, var], 2)
                if self.topology[1] == "var":
                    cat = torch.cat([cat, sqr], 2)
            if len(self.topology) == 3:
                cat = torch.cat([x, mean, var, sqr], 2)
        else:
            cat = x
        ula, (h_out, _) = self.lstm(cat, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        if self.num_linear_layers == 1:
            out = self.fc1(h_out)
        if self.num_linear_layers == 2:
            out = self.fc2(h_out)
            out = self.fc1(out)
        if self.num_linear_layers == 3:
            out = self.fc3(h_out)
            out = self.fc2(out)
            out = self.fc1(out)
        if self.num_linear_layers == 4:
            out = self.fc4(h_out)
            out = self.fc3(out)
            out = self.fc2(out)
            out = self.fc1(out)
        if self.num_linear_layers == 0:
            out = h_out
        out = out[-self.batch_size:]

        return out

    def training_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        out = out[-self.batch_size:]
        if self.reg == 'L1':
            l1_lambda = self.lmd
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss = self.loss_fcn(out, labels) + l1_lambda * l1_norm
        if self.reg == 'L2':
            l2_lambda = self.lmd
            l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = self.loss_fcn(out, labels) + l2_lambda * l2_norm
        if self.reg == 'none':
            loss = self.loss_fcn(out, labels)  # Calculate loss
        # loss = self.loss_fcn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return acc, loss

    def validation_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        loss = self.loss_fcn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        # epoch_acc = 0
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        # return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}]: train_acc: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_acc'], result['train_loss'], result['val_loss'], result['val_acc']))
