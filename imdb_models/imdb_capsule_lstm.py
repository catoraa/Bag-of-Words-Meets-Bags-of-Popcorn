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

num_epochs = 10
embed_size = 300
num_hiddens = 128
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.01
device = torch.device('cuda:0')
use_gpu = True


class Capsule(nn.Module):
    def __init__(self, num_hiddens, bidirectional, num_capsule=5, dim_capsule=5, routings=4, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.bidirectional = bidirectional

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = self.squash

        if self.bidirectional:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.num_hiddens * 2, self.num_capsule * self.num_hiddens * 2)))
        else:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.num_hiddens, self.num_capsule * self.num_hiddens)))

    def forward(self, inputs):
        #print(inputs.shape)
        #print(self.W.shape)
        u_hat_vecs = torch.matmul(inputs, self.W)
        batch_size = inputs.size(0)
        input_num_capsule = inputs.size(1)
        #print(u_hat_vecs.shape)
        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule, self.num_capsule, -1)

        u_hat_vecs = u_hat_vecs.permute(
            0, 2, 1, 3).contiguous()  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        with torch.no_grad():
            b = torch.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = torch.nn.functional.softmax(b, dim=1)  # (batch_size,num_capsule,input_num_capsule)
            outputs = self.activation(torch.sum(c.unsqueeze(-1) * u_hat_vecs, dim=2))  # bij,bijk->bik
            if i < self.routings - 1:
                b = (torch.sum(outputs.unsqueeze(2) * u_hat_vecs, dim=-1))  # bik,bijk->bij
        return outputs  # (batch_size, num_capsule, dim_capsule)

    @staticmethod
    def squash(x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale


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
        # self.attention = Attention(num_hiddens=self.num_hiddens, bidirectional=self.bidirectional)
        self.capsule = Capsule(num_hiddens=self.num_hiddens, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # 获取词嵌入
        states, hidden = self.encoder(embeddings.permute(1, 0, 2))  # LSTM 编码
        states = states.permute(1, 0, 2)
        capsule = self.capsule(states)  # Capsule 网络

        # 问题修正，调整了取capsule的顺序，以适配tensor size的不一致
        encoding = torch.cat([capsule[:, 0, :], capsule[:, -1, :]], dim=1)  # 拼接 Capsule 输出

        outputs = self.decoder(encoding)  # 解码得到分类结果
        return outputs  # 返回最终的输出


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
                train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
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
    result_output.to_csv("../result/capsule_lstm.csv", index=False, quoting=3)
    logging.info('result saved!')
