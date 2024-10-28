import os
import sys
import logging
import datasets
import evaluate
from transformers.modeling_outputs import SequenceClassifierOutput

import losses

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments,DebertaForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

import torch.nn as nn

train = pd.read_csv("/kaggle/input/bag-of-word/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/bag-of-word/testData.tsv", header=0, delimiter="\t", quoting=3)


class CustomModelForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config, alpha=0.5):
        super().__init__(config)
        self.alpha = alpha

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        pooled_output = logits  # 假设 pooled_output 是第一个返回的输出

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            scl_fct = losses.SupConLoss()
            scl_loss = scl_fct(pooled_output, labels)

            loss = ce_loss + self.alpha * scl_loss

        return (loss, logits) if loss is not None else logits


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = "e1a47aca16f2292eb9d8fe1d613c1ac623dd63a6"
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
        return tokenizer(examples['text'], truncation=True,padding='max_length', max_length=510)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_id,)

    # 设定LoRAConfig
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        #target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # 添加LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("/kaggle/working/deberta_lora.csv", index=False, quoting=3)
    logging.info('result saved!')
