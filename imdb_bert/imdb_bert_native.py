import os
import sys
import logging
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


train = pd.read_csv("../test_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../test_data/testData.tsv", header=0, delimiter="\t", quoting=3)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, num_samples=0):
        self.encodings = encodings
        self.num_samples = num_samples

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optim = optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        with tqdm(total=len(train_loader), desc="Epoch %d" % epoch) as pbar:
            for batch in train_loader:
                n += 1
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optim.step()
                train_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                train_loss += loss.cpu()

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)

            with torch.no_grad():
                for batch in val_loader:
                    m += 1
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss
                    val_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)})

            # print('epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Predction') as Pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                # test_pred.extent
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/bert_native.csv", index=False, quoting=3)
    logging.info('result saved!')
