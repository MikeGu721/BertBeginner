from models.BERT import BertModel
from pytorch_transformers import BertTokenizer
from preprocess.SentenceClassification import LoadSingleSentenceClassificationDataset
import Config
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os


class BertForSentenceClassification(nn.Module):
    def __init__(self, config, pretrained_model_dir=None):
        super(BertForSentenceClassification, self).__init__()
        self.num_labels = config.num_labels
        if pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(model_config, model_config.pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_id=None, position_ids=None, labels=None):
        pooled_output, _ = self.bert(input_ids, attention_mask, token_type_id, position_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


def train(model_config, dataset_name):
    # 处理数据
    data_loader = LoadSingleSentenceClassificationDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )
    train_iter, val_iter, test_iter, label_size = data_loader.load_train_val_test_data(
        model_config.train_file_path,
        model_config.val_file_path)
    # 刷新label数量
    model_config.num_labels = label_size
    # 读取模型
    model = BertForSentenceClassification(model_config,
                                          model_config.pretrained_model_dir)
    if not os.path.exists(os.path.join(model_config.pretrained_model_dir, 'BertForSentenceClassification')):
        os.makedirs(os.path.join(model_config.pretrained_model_dir, 'BertForSentenceClassification'))
    model_save_path = os.path.join(model_config.pretrained_model_dir, 'BertForSentenceClassification',
                                   dataset_name + '_pt')
    model.to(model_config.device)
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
    model.train()

    # 开始训练
    max_acc = 0
    for epoch in range(model_config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (data, label) in enumerate(train_iter):
            data = data.to(model_config.device)
            label = label.to(model_config.device)
            padding_mask = (data == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(input_ids=data, attention_mask=padding_mask, labels=label)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                logging.info(f'Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], '
                             f'Train loss: {loss.item():.4f}, Train acc: {acc:.4f}')
        end_time = time.time()
        logging.info(f'Epoch: {epoch}, Train loss: {losses:.3f}, Epoch Time: {end_time - start_time:.4f}')
        if (epoch + 1) % model_config.model_save_per_epoch == 0:
            acc = evaluate(val_iter, model, model_config.device)
            logging.info(f'Accuracy on val {acc:4f}')
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)
    logging.info(f'Highest Accuracy on Val {max_acc:4f}')


def evaluate(data_iter, model, device, PAD_IDX=0):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0, 0
        for (data, label) in data_iter:
            data, label = data.to(device), label.to(device)
            padding_mask = (data == PAD_IDX).transpose(0, 1)
            logits = model(data, attention_mask=padding_mask)
            acc_sum += (logits.argmax(1) == label).float().sum().item()
            n += len(label)
    model.train()
    return acc_sum / n


if __name__ == '__main__':
    model_config = Config.ModelConfig(bert_name='bert-base-chinese', dataset_name='SogouNews-Chinese')
    model_config.learning_rate = 1e-5
    train(model_config, dataset_name='SogouNews-Chinese')
