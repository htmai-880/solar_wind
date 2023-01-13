from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import gc

import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return compute_rolling_std(X, 'Beta', '2h')


# Transformer NN
class TransformerModel(nn.Module):
    def __init__(self, n_in, nhead, nhid, nlayers, dropout=0.2):
        super(TransformerModel, self).__init__()
        '''
        n_in: dimension of input
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        self.fc = nn.Linear(n_in, nhid)
        self.pos_encoder = PositionalEncoding(nhid=nhid, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            nhid, nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=nlayers)
        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.fc(src) * np.sqrt(self.nhid)
        src = self.pos_encoder(src)  # fill me
        output = self.transformer_encoder(src, mask=src_mask)  # fill me
        return output


class ClassificationHead(nn.Module):
    def __init__(self, nhid, nclasses):
        super(ClassificationHead, self).__init__()
        self.decoder = nn.Linear(nhid, nclasses)  # fill me)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.decoder(src)
        return output


class Model(nn.Module):
    def __init__(self, n_in, nhead, nhid, nlayers, nclasses, dropout=0.2):
        super(Model, self).__init__()
        self.base = TransformerModel(
            n_in, nhead, nhid, nlayers, dropout=dropout)
        self.classifier = ClassificationHead(nhid, nclasses)

    def forward(self, src, src_mask):
        # base model
        x = self.base(src, src_mask)
        # classifier model
        output = self.classifier(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-np.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

#### Data loaders ####


class MyDataset(Dataset):
    def __init__(self, X, y=None, max_len=64):
        self.X = X
        self.y = y
        if not y is None:
            # print("Built dataset. len y:", len(y))
            self.max_history = max_len-1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Get data (x, y)

        start = max(0, index-self.max_history)
        # sequence includes X.iloc[index]
        sequence = self.X.iloc[start:index+1]
        # print("index: ", index)
        # print("Sequence :\n", sequence)
        sequence = torch.tensor(sequence.values)
        if not self.y is None:
            # print(self.y)
            label = self.y[index]
            # print("label : ", label)
            label = torch.tensor([label], dtype=torch.int64)
            return sequence, label
        else:
            return (sequence,)


def MyCollator(batch):
    """This function pads the input with zeros so that it has matching sequence length."""
    # print("batch item = \n", batch[0][0].size())
    source_sequences = pad_sequence([sample[0] for sample in batch])
    # print("label item = ", batch[0][1])
    if len(batch[0]) == 1:
        return (source_sequences,)

    labels = pad_sequence([torch.tensor([sample[1]]) for sample in batch])
    return source_sequences, labels.reshape(-1)


def get_loader(X, y=None, max_len=64, batch_size=32):
    dataset = MyDataset(X, y, max_len=max_len)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             collate_fn=MyCollator, pin_memory=True, drop_last=True)
    return data_loader

##########################################################

# Train step


def train_batch(model, data, optimizer, criterion):
    params = next(model.parameters())
    device = params.device

    batch_loss = 0

    optimizer.zero_grad()
    src_mask = model.base.generate_square_subsequent_mask(
        data[0].size(0)).to(device)
    input_ = data[0].float().to(device)
    output = model(input_, src_mask)
    # last vector only
    output = output[-1]
    output = output.view(-1, output.shape[-1]).float().to(device)
    target = data[1]
    target = target.to(device)
    loss = criterion(output, target)  # Cross entropy check next cells

    loss.backward()
    # prevent exploding gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    batch_loss += loss.item()

    # cleanup memory
    del output, target
    gc.collect()
    if params.is_cuda:
        torch.cuda.empty_cache()

    return batch_loss


def train_epoch(model, X_train, y_train, optimizer, criterion, max_len=64, log_interval=5, batch_size=32):
    model.train()
    total_loss = 0
    # print(y_train)
    data_loader = get_loader(
        X_train, y_train, max_len=max_len, batch_size=batch_size)
    losses = []
    for idx, data in enumerate(data_loader):
        # print("--- New batch ---")
        # print("len y in this batch: ", len(y_train))
        # print("idx: ", idx)
        # print(data[0].size())
        # print(data[1].size())
        batch_loss = train_batch(model, data, optimizer, criterion)
        total_loss += batch_loss
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            print(
                "| {:5d}/{:5d} steps | "
                "loss {:5.5f} | ppl {:8.3f}".format(
                    idx, len(data_loader), cur_loss, np.exp(cur_loss),
                )
            )
            losses.append(cur_loss)
            total_loss = 0

        # print("--- Finished batch ---")
    return losses

##########################################################


class Classifier(BaseEstimator):

    def __init__(self, nhead, nhid, nlayers, nclasses=2, dropout=0.2, max_len=64, epochs=5, batch_size=16, lr=1e-3):
        self.n_in = None
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.nclasses = nclasses
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.optimizer = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Device : ", self.device)

    def fit(self, X, y):
        self.n_in = X.shape[1]
        self.model = Model(self.n_in, self.nhead, self.nhid,
                           self.nlayers, self.nclasses, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        
        # X = X.iloc[:100]
        # y = y.iloc[:100]

        log_interval = 50
        for epoch in range(1, self.epochs+1):
            print("----------- Epoch " + str(epoch) + " -----------")
            train_epoch(self.model, X, y, self.optimizer, self.criterion,
                        max_len=self.max_len, log_interval=log_interval, batch_size=self.batch_size)
            torch.save(self.model.state_dict(), './best_model.pt')


    def predict(self, X):
        self.model.eval()
        get_loader(X, None, max_len=self.max_len, batch_size=self.batch_size)
        y = []
        data_loader = get_loader(
            X, y=None, max_len=self.max_len, batch_size=self.batch_size)
        for idx, data in enumerate(data_loader):
            src_mask = self.model.base.generate_square_subsequent_mask(
                data[0].size(0)).to(self.device)
            input_ = data[0].float().to(self.device)
            y.append(self.model(input_, src_mask))
        y_pred = torch.cat(y, dim=0)
        return y_pred.numpy()


def get_estimator():

    nhead = 4
    nhid = 128
    nlayers = 3
    dropout = 0.3
    epochs = 2
    batch_size = 8
    lr = 1e-3

    feature_extractor = FeatureExtractor()

    classifier = Classifier(nhead, nhid, nlayers, nclasses=2, dropout=dropout, max_len=64, epochs=epochs, batch_size=batch_size, lr=lr)

    pipe = make_pipeline(feature_extractor, classifier)
    return pipe


def compute_rolling_std(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
