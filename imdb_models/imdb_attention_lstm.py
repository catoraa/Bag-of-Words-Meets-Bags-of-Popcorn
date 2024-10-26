import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm

from sklearn.metrics import accuracy_score

test = pd.read_csv("../test_data/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10 # 迭代次数
embed_size = 300  # 嵌入层大小
num_hiddens = 128  # 隐藏层大小
num_layers = 2  # LSTM单元数量
bidirectional = True  # 是否双向
batch_size = 64 # batch_size大小
labels = 2  # label列数
lr = 0.01  # 初始学习率
device = torch.device('cuda:0')  # 设定计算设备为显卡
use_gpu = True  # 设定计算设备为显卡


class Attention(nn.Module):
    def __init__(self, num_hiddens, bidirectional, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.bidirectional = bidirectional

        # if bidirectional, then double the hidden dimensionality
        if self.bidirectional:
            self.w_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, num_hiddens * 2))
            self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        else:
            self.w_omega = nn.Parameter(torch.Tensor(num_hiddens, num_hiddens))
            self.u_omega = nn.Parameter(torch.Tensor(num_hiddens, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        x = inputs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)

        att_score = F.softmax(att, dim=1)
        outputs = x * att_score
        return outputs


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        self.attention = Attention(num_hiddens=self.num_hiddens, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute(1, 0, 2))
        attention = self.attention(states)
        encoding = torch.cat([attention[0], attention[-1]], dim=1)
        outputs = self.decoder(encoding)
        # print(outputs)
        return outputs


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = 'imdb_glove.pickle3'
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                net.zero_grad()
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                score = net(feature)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                train_loss += loss

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)

            with torch.no_grad():
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
                    val_label = val_label.cuda()
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)
                              })

            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.cuda()
                test_score = net(test_feature)
                # test_pred.extent
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/attention_lstm.csv", index=False, quoting=3)
    logging.info('result saved!')
