
文件夹内容如下：<br>
imdb_models:提供的imdb预测模型，已修改以适配新版本的库<br>
my_code:kaggle教程练习代码<br>
result:各模型跑出的预测数据<br>
test_data:测试数据集<br>
<br>
<br>
模型结果如下：

| 模型                 | 训练准确率   | 预测准确率   | 备注                         |
|:-------------------|:--------|:--------|:---------------------------|
| Attention_LSTM     | 0.93    | 0.86    | 速度较快                       |
| BERT_Native        | 0.96    | 0.91    | 速度非常慢，显存占用较高               |
| BERT_Scratch       | /       | /       | 适配了最新的库，跑不出来               |
| BERT_Trainer       | /       | /       | 适配了最新的库，跑不出来               |
| Capsule_LSTM       | /       | /       | input和target的batch_size不匹配 |
| CNN                | 0.75    | 0.75    | 改为跑20轮，效果有一定提升             |
| CNN_LSTM           | 0.81    | 0.78    | 比起纯CNN效果有一定提升              |
| DistilBERT_Native  | 0.97    | 0.91    | 速度非常慢，显存占用非常高              |
| DistilBERT_Trainer | 0.93    | 0.93    | 适配了最新的库，kaggle跑出来了         |
| GRU                | 0.86    | 0.86    | 速度较快，效果还行                  |
| LSTM               | 0.75    | 0.75    | 速度较快，效果一般                  |
| RoBERTa_Trainer    | /       | /       | nan                        |
| Transformer        | /       | /       | size of tensor不匹配          |
| BoW+RF             | /       | 0.85    | 速度较快，效果还行                  |
| Word2Vec+RF        | /       | 0.83    | 速度较快，效果还行                  |