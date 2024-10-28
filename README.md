
文件夹内容如下：<br>
imdb_bert:bert类的imdb预测模型<br>
imdb_deberta:deberta类的imdb预测模型<br>
imdb_models:未归类的imdb预测模型<br>
my_code:kaggle教程练习代码<br>
result:各模型跑出的预测数据<br>
test_data:测试数据集<br>
<br>
<br>
模型结果如下：

| 模型                 | 训练准确率 | 预测准确率 | 备注                             |
|:-------------------|:------|:------|:-------------------------------|
| Attention_LSTM     | 93%   | 87%   | 速度较快                           |
| BERT_Native        | 96%   | 91%   | 速度非常慢，显存占用较高                   |
| BERT_Scratch       | 93%   | 93%   | 修正前向传播时labels参数调用逻辑问题          |
| BERT_Trainer       | 94%   | 94%   | 适配最新版本库，kaggle跑出来了             |
| Capsule_LSTM       | 89%   | 88%   | 修正torch.cat前的tensor shape不匹配问题 |
| CNN                | 75%   | 75%   | 改为跑20轮，效果有一定提升                 |
| CNN_LSTM           | 81%   | 78%   | 比起纯CNN效果有一定提升                  |
| DistilBERT_Native  | 97%   | 91%   | 速度非常慢，显存占用非常高                  |
| DistilBERT_Trainer | 93%   | 93%   | 适配最新版本库，kaggle跑出来了             |
| GRU                | 86%   | 86%   | 速度较快，效果还行                      |
| LSTM               | 75%   | 75%   | 速度较快，效果一般                      |
| RoBERTa_Trainer    | 95%   | 95%   | 适配了最新的库，kaggle跑出来了             |
| Transformer        | /     | /     | transformer的前向传播缺少lengths参数?   |
| BoW+RF             | /     | 85%   | 速度较快，效果还行                      |
| Word2Vec+RF        | /     | 83%   | 速度较快，效果还行                      |
| Deberta_lora       | 97%   | 97%   | 速度慢，效果全场最佳                     |
| Deberta_prompt     | 71%   | 71%   | 速度慢，效果一般                       |
| Deberta_prefix     | /     | /     | 模型不兼容                          |
| Deberta_ptuning    | 58%   | 58%   | 速度慢，效果差                        |
| BERT_rdrop         | 93%   | 94%   | 速度慢，效果不错                       |
| BERT_scl           | 93%   | 94%   | 速度慢，效果不错                       |
