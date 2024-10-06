import re
import pandas as pd
import nltk.data
import logging
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


# 训练数据预处理
train = pd.read_csv("./test_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./test_data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./test_data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews,and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size))

# 准备数据
sentences = []  # Initialize an empty list of sentences
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# 训练模型
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count=min_word_count,window=context, sample=downsampling)
model.init_sims(replace=True)

# 模型存储
model_name = "300features_40minwords_10context"
model.save(model_name)
