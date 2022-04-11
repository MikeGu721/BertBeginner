import torch
import torch.nn as nn
import torch.optim as optim
from models.BERT import BertModel

from preprocess import MachineReadingComprehansion
from pytorch_transformers import BertTokenizer
import os
import time
import logging
import collections


class BertForMRC(nn.Module):
    def __init__(self, config, pretrained_model_dir=None):
        super(BertForMRC, self).__init__()
        if pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(model_config, model_config.pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_id=None, position_ids=None, start_position=None,
                end_position=None):
        _, all_encoder_outputs = self.bert(input_ids, attention_mask, token_type_id, position_ids)
        # 取bert最后一层的输出
        sequence_output = all_encoder_outputs[-1]  # [src_len, batch_size, emb_size]
        logits = self.qa_outputs(sequence_output)  # [src_len, batch_size, 2]
        start_logits, end_logits = logits.split(1, dim=-1)  # [src_len, batch_size], [src_len, batch_size]
        start_logits = start_logits.squeeze(-1).transpose(0, 1)  # [batch_size, src_len]
        end_logits = end_logits.squeeze(-1).transpose(0, 1)  # [batch_size, src_len]
        if start_position is not None and end_position is not None:
            ignore_index = start_logits.size(1)
            start_position.clamp_(0, ignore_index)  # 强制截断
            end_position.clamp_(0, ignore_index)  # 强制截断 保证src_len <= ignore_index

            # 这种计算损失函数的方式似乎有很大的改进空间
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            start_loss = loss_fct(start_logits, start_position)
            end_loss = loss_fct(end_logits, end_position)
            return (start_loss + end_loss) / 2, start_logits, end_logits
        else:
            return start_logits, end_logits


def show_result(batch_input, itos, num_show=5, y_pred=None, y_true=None):
    count = 0
    batch_input = batch_input.transpose(0, 1)
    for i in range(len(batch_input)):
        if count == num_show:
            break
        input_tokens = [itos[s] for s in batch_input[i]]
        start_pos, end_pos = y_pred[0][i], y_pred[1][i]
        answer_text = ' '.join(input_tokens[start_pos:(end_pos + 1)]).replace(' ##', '')
        input_text = ' '.join(input_tokens).replace(' ##', '').split('[SEP]')
        question_text, context_text = input_text[0], input_text[1]

        logging.info(f'### Question: {question_text}')
        logging.info(f' ## Predicted answer: {answer_text}')
        start_pos, end_pos = y_true[0][i], y_true[1][i]
        true_answer_text = ' '.join(input_tokens[start_pos:(end_pos + 1)])
        true_answer_text = true_answer_text.replace(" ##", '')
        logging.info(f' ## True answer: {true_answer_text}')
        logging.info(f' ## True answer idx: {start_pos.cpu(), end_pos.cpu()}')
        count += 1


def train(model_config, dataset_name):
    # 处理数据
    data_loader = MachineReadingComprehansion.LoadSQuADQuestionAnsweringDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )
    train_iter, val_iter, test_iter, test_examples = data_loader.load_train_val_test_data(
        model_config.train_file_path,
        model_config.val_file_path)
    # 读取模型
    model = BertForMRC(model_config, model_config.pretrained_model_dir)
    if not os.path.exists(os.path.join(model_config.pretrained_model_dir, 'BertForMRC')):
        os.makedirs(os.path.join(model_config.pretrained_model_dir, 'BertForMRC'))
    model_save_path = os.path.join(model_config.pretrained_model_dir, 'BertForMRC', dataset_name + '_pt')
    model.to(model_config.device)
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
    model.train()

    # 开始训练
    max_acc = 0
    for epoch in range(model_config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (data, seg, label, _, _, _, _) in enumerate(train_iter):
            data = data.to(model_config.device)
            seg = seg.to(model_config.device)
            label = label.to(model_config.device)
            padding_mask = (data == data_loader.PAD_IDX).transpose(0, 1)
            loss, start_logits, end_logits = model(input_ids=data, attention_mask=padding_mask, token_type_id=seg,
                                                   position_ids=None,
                                                   start_position=label[:, 0], end_position=label[:, 1])
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_start = (start_logits.argmax(1) == label[:, 0]).float().mean()
            acc_end = (end_logits.argmax(1) == label[:, 1]).float().mean()
            acc = (acc_start + acc_end) / 2
            if idx % 10 == 0:
                logging.info(f'Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], '
                             f'Train loss: {loss.item():.4f}, Train acc: {acc:.4f}')
            if idx % 50 == 0 and idx != 0:
                y_pred = [start_logits.argmax(1), end_logits.argmax(1)]
                y_true = [label[:, 0], label[:, 1]]
                show_result(data, data_loader.vocab.itos, y_pred=y_pred, y_true=y_true)

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
        all_results = collections.defaultdict(list)
        for (data, seg, label, qid, _, feature_id, _) in data_iter:
            data, seg, label = data.to(device), seg.to(device), label.to(device)
            padding_mask = (data == PAD_IDX).transpose(0, 1)
            start_logits, end_logits = model(data, attention_mask=padding_mask, token_type_id=seg)
            all_results[qid[0]].append(
                [feature_id[0], start_logits.cpu().numpy().reshape(-1), end_logits.cpu().numpy().reshape(-1)])
            acc_sum_start = (start_logits.argmax(1) == label[:, 0]).float().sum().mean()
            acc_sum_end = (end_logits.argmax(1) == label[:, 1]).float().sum().mean()
            acc_sum += acc_sum_end + acc_sum_start
            n += len(label)
    model.train()
    return acc_sum / (2 * n)


if __name__ == '__main__':
    from Config import ModelConfig

    bert = 'bert-base-chinese'
    dataset = 'SQuAD-Chinese'
    model_config = ModelConfig(bert_name=bert, dataset_name=dataset, task='MRC')
    model_config.test_file_path = None
    model_config.train_file_path = model_config.train_file_path.split('.txt')[0] + '.json'
    model_config.val_file_path = model_config.val_file_path.split('.txt')[0] + '.json'
    model_config.epochs = 100
    model_config.learning_rate = 1e-4
    train(model_config, dataset_name=dataset)
