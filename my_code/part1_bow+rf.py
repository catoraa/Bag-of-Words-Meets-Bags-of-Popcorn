import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


# 训练数据预处理
train = pd.read_csv("../test_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = train["review"].size
clean_train_reviews = []
print("Cleaning and parsing the train set movie reviews...")
for i in range(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

# 从单词袋中创建特征
print("Creating the bag of words...")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# 随机森林分类
print("Training the random forest...")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

# 预测数据预处理
test = pd.read_csv("../test_data/testData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing the test set movie reviews...")
for i in range(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d" % (i + 1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# 随机森林预测
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print("Predict with random forest...")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("../result/BoW_RF.csv", index=False, quoting=3)
print("Result saved!")
