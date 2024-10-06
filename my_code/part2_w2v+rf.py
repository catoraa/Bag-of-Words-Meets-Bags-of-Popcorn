import re
import pandas as pd
import nltk.data
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review,"lxml").get_text()
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
train = pd.read_csv("../test_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../test_data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("../test_data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print("Read %d labeled train reviews, %d labeled test reviews,and %d unlabeled reviews" % (
    train["review"].size, test["review"].size, unlabeled_train["review"].size))

# 准备数据
sentences = []  # Initialize an empty list of sentences
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

# word2vec模型训练
num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features,
                          min_count=min_word_count, window=context, sample=downsampling)

# 保存模型
model_name = "300features_40minwords_10context"
model.save(model_name)


# word2vec模型编码
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # 使用 model.wv.index_to_key 来获取词汇表
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])  # 修改为 model.wv[word]
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


# 创建平均特征向量
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# 训练随机森林
forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# 预测结果
result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("../result/Word2Vec_RF.csv", index=False, quoting=3)
print("Result saved!")
