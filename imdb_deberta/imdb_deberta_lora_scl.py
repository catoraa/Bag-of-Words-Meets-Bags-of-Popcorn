import logging
import os
import sys
from typing import Optional, Tuple, Union

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['WANDB_API_KEY'] = "e1a47aca16f2292eb9d8fe1d613c1ac623dd63a6"
sys.path.append("../input")

import torch
import datasets
import evaluate
import numpy as np
import pandas as pd
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DebertaV2Tokenizer, DataCollatorWithPadding, DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout

import losses

train = pd.read_csv("/kaggle/input/bag-of-word/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/bag-of-word/testData.tsv", header=0, delimiter="\t", quoting=3)


class DebertaLoraScl(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = 0.2
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            scl_fct = losses.SupConLoss()
            scl_loss = scl_fct(pooled_output, labels)

            loss = ce_loss + self.alpha * scl_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    model_id = "microsoft/deberta-v3-large"

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=510)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DebertaLoraScl.from_pretrained(model_id)

    # 设定LoRAConfig
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # 此处不用手动设定target_modules，而是由lora自动适配
        #target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # 加载LoRA到模型上
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric = evaluate.load("accuracy")

    # 训练指标计算
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 训练过程可视化
    training_args = TrainingArguments(
        output_dir='./deberta_lora_scl',  # 输出文件夹，即wandb上显示的名字
        num_train_epochs=3,  # 迭代次数
        per_device_train_batch_size=2,  # 训练batch_size
        per_device_eval_batch_size=4,  # 评估batch_size
        warmup_steps=500,  # 学习率热身步数
        weight_decay=0.01,  # 权重衰减系数
        logging_dir='./logs',  # logs存储目录
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )
    # 训练参数
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # 调用trainer训练模型
    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("/kaggle/working/deberta_lora.csv", index=False, quoting=3)
    logging.info('result saved!')
