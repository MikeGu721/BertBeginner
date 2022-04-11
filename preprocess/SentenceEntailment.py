from utils import pad_sequence
import tqdm
import torch
import gzh
from BertForSentenceClassification import LoadSingleSentenceClassificationDataset


class LoadPairSentenceClassificationDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, **kwargs):
        super(LoadPairSentenceClassificationDataset, self).__init__(**kwargs)
        pass

    def data_process(self, filepath):
        raw_iter = open(filepath, encoding='utf-8')
        data = []
        max_len = 0
        label2id = {}
        index = 0
        for raw in tqdm.tqdm(raw_iter, ncols=80):
            line = raw.rstrip('\n').split(self.split_sep)
            s1, s2, l = line
            if l not in label2id:
                label2id[l] = len(label2id)
                l = label2id[l]
            token1 = [self.vocab[token] for token in self.tokenizer.tokenize(s1)]
            token2 = [self.vocab[token] for token in self.tokenizer.tokenize(s2)]
            tmp = [self.CLS_IDX] + token1 + [self.SEP_IDX] + token2
            if len(tmp) > self.max_position_embedding - 1:
                tmp = tmp[:self.max_position_embedding - 1]
            tmp += [self.SEP_IDX]
            seg1 = [0] * (len(token1) + 2)
            seg2 = [1] * (len(tmp) - len(seg1))
            segs = torch.tensor(seg1 + seg2, dtype=torch.long)
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, segs, l))
            # index += 1
            # if index == 100:
            #     break
        return data, max_len, len(label2id)

    def generate_batch(self, data_batch):
        batch_sentence, batch_seg, batch_label = [], [], []
        for (sen, seg, l) in data_batch:
            batch_sentence.append(sen)
            batch_seg.append(seg)
            batch_label.append(l)
        batch_sentence = pad_sequence(batch_sentence, padding_value=self.PAD_IDX,
                                      batch_first=False, max_len=self.max_sen_len)
        batch_seg = pad_sequence(batch_seg, padding_value=self.PAD_IDX,
                                 batch_first=False, max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_seg, batch_label


if __name__ == '__main__':
    import Config
    from pytorch_transformers import BertTokenizer

    model_config = Config.ModelConfig(bert_name='bert-base-uncased', dataset_name='multinli', task='Entailment')
    model_config.split_sep = '_!_'
    load_dataset = LoadPairSentenceClassificationDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )
    train_iter, val_iter, test_iter, label_size = load_dataset.load_train_val_test_data(None, None,
                                                                                        model_config.test_file_path)
    test_data,_,_ = load_dataset.data_process(model_config.test_file_path)
    for sample, seg, label in test_data:
        print(sample.shape)
        print(seg.shape)
        print(label.shape)
        break
    for sample, seg, label in test_iter:
        print(sample)
        print(sample.shape)
        print(seg.shape)
        print(label.shape)
        break
