
文件夹内容如下：<br>
imdb_models:提供的imdb预测模型，已修改以适配新版本的库<br>
my_code:kaggle教程练习代码<br>
result:各模型跑出的预测数据<br>
test_data:测试数据集<br>
<br>
<br>
模型结果如下：

| 模型                 | 训练准确率 | 预测准确率 | 备注                             |
|:-------------------|:------|:------|:-------------------------------|
| Attention_LSTM     | 0.93  | 0.87  | 速度较快                           |
| BERT_Native        | 0.96  | 0.91  | 速度非常慢，显存占用较高                   |
| BERT_Scratch       | 0.93  | 0.93  | 修正前向传播时labels参数调用逻辑问题          |
| BERT_Trainer       | 0.94  | 0.94  | 适配最新版本库，kaggle跑出来了             |
| Capsule_LSTM       | 0.89  | 0.88  | 修正torch.cat前的tensor shape不匹配问题 |
| CNN                | 0.75  | 0.75  | 改为跑20轮，效果有一定提升                 |
| CNN_LSTM           | 0.81  | 0.78  | 比起纯CNN效果有一定提升                  |
| DistilBERT_Native  | 0.97  | 0.91  | 速度非常慢，显存占用非常高                  |
| DistilBERT_Trainer | 0.93  | 0.93  | 适配最新版本库，kaggle跑出来了             |
| GRU                | 0.86  | 0.86  | 速度较快，效果还行                      |
| LSTM               | 0.75  | 0.75  | 速度较快，效果一般                      |
| RoBERTa_Trainer    | 0.95  | 0.95  | 适配了最新的库，kaggle跑出来了             |
| Transformer        | /     | /     | transformer的前向传播缺少lengths参数?   |
| BoW+RF             | /     | 0.85  | 速度较快，效果还行                      |
| Word2Vec+RF        | /     | 0.83  | 速度较快，效果还行                      |
| deberta_lora       | 0.94  | 0.94  | 速度还行，效果不错                      |