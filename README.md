
�ļ����������£�<br>
imdb_bert:bert���imdbԤ��ģ��<br>
imdb_deberta:deberta���imdbԤ��ģ��<br>
imdb_models:δ�����imdbԤ��ģ��<br>
my_code:kaggle�̳���ϰ����<br>
result:��ģ���ܳ���Ԥ������<br>
test_data:�������ݼ�<br>
<br>
<br>
ģ�ͽ�����£�

| ģ��                 | ѵ��׼ȷ�� | Ԥ��׼ȷ�� | ��ע                             |
|:-------------------|:------|:------|:-------------------------------|
| Attention_LSTM     | 93%   | 87%   | �ٶȽϿ�                           |
| BERT_Native        | 96%   | 91%   | �ٶȷǳ������Դ�ռ�ýϸ�                   |
| BERT_Scratch       | 93%   | 93%   | ����ǰ�򴫲�ʱlabels���������߼�����          |
| BERT_Trainer       | 94%   | 94%   | �������°汾�⣬kaggle�ܳ�����             |
| Capsule_LSTM       | 89%   | 88%   | ����torch.catǰ��tensor shape��ƥ������ |
| CNN                | 75%   | 75%   | ��Ϊ��20�֣�Ч����һ������                 |
| CNN_LSTM           | 81%   | 78%   | ����CNNЧ����һ������                  |
| DistilBERT_Native  | 97%   | 91%   | �ٶȷǳ������Դ�ռ�÷ǳ���                  |
| DistilBERT_Trainer | 93%   | 93%   | �������°汾�⣬kaggle�ܳ�����             |
| GRU                | 86%   | 86%   | �ٶȽϿ죬Ч������                      |
| LSTM               | 75%   | 75%   | �ٶȽϿ죬Ч��һ��                      |
| RoBERTa_Trainer    | 95%   | 95%   | ���������µĿ⣬kaggle�ܳ�����             |
| Transformer        | /     | /     | transformer��ǰ�򴫲�ȱ��lengths����?   |
| BoW+RF             | /     | 85%   | �ٶȽϿ죬Ч������                      |
| Word2Vec+RF        | /     | 83%   | �ٶȽϿ죬Ч������                      |
| Deberta_lora       | 97%   | 97%   | �ٶ�����Ч��ȫ�����                     |
| Deberta_prompt     | 71%   | 71%   | �ٶ�����Ч��һ��                       |
| Deberta_prefix     | /     | /     | ģ�Ͳ�����                          |
| Deberta_ptuning    | 58%   | 58%   | �ٶ�����Ч����                        |
| BERT_rdrop         | 93%   | 94%   | �ٶ�����Ч������                       |
| BERT_scl           | 93%   | 94%   | �ٶ�����Ч������                       |
