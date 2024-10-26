import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from sklearn.metrics import accuracy_score

test = pd.read_csv("../test_data/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 20  # 迭代次数
embed_size = 300  # 嵌入层大小
num_filter = 128
filter_size = 3  #
bidirectional = True  # 双向
batch_size = 64
labels = 2
lr = 0.8  #学习率
device = torch.device('cuda:0')
use_gpu = True


class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_filter, filter_size, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.use_gpu = use_gpu
        self.embedding = nn.Embedding.from_pretrained(weight)  # 嵌入层
        self.embedding.weight.requires_grad = False

        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.decoder = nn.Linear(num_filter, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)

        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))  # 卷积层
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])  # 池化层
        outputs = self.decoder(pooling.squeeze(dim=2)) # 解码输出层
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

    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    net.to(device) # 将网络转移到GPU
    loss_function = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 可视化训练过程
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
                              'time': '%.2f' % (runtime)})

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
    result_output.to_csv("../result/cnn.csv", index=False, quoting=3)
    logging.info('result saved!')
