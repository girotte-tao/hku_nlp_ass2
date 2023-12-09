from __future__ import absolute_import, division, print_function
import logging
import torch.nn.functional as F
from transformers import AdamW, AutoTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from LBNER_win import LBNER

import logging
import os
from datetime import datetime

#  ----- log config -----
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, domain_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain_label = domain_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, domain_label=None,
                 seq_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.domain_label = domain_label
        self.seq_len = seq_len


def readfile(filename, type_=None):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                assert len(sentence) == len(label)
                data.append((sentence, label))
                sentence = []
                label = []
            continue

        # TODO
        if type_ != 'predict':
            splits = line.strip().split()
            # print(splits)
            sentence.append(splits[0])
            label.append(splits[-1])
            # domain_l = eval(splits[2])

        else:
            splits = line.strip().split()
            sentence.append(splits[0])
            label.append('O')

    if len(sentence) > 0:
        data.append((sentence, label))
        assert len(sentence) == len(label)
        sentence = []
        label = []
    return data


def readfile_label(train_file, test_file, dev_file):
    '''
    read file
    '''
    label_set = set()
    f_train = open(train_file)
    f_test = open(test_file)
    f_dev = open(dev_file)

    for line in f_train:
        temp = line.strip()
        if temp != '':
            splits = line.strip().split()
            if splits[-1] != 'O':
                label_set.add(splits[-1])

    for line in f_test:
        temp = line.strip()
        if temp != '':

            splits = line.strip().split()
            if splits[-1] != 'O':
                label_set.add(splits[-1])

    for line in f_dev:
        temp = line.strip()
        if temp != '':

            splits = line.strip().split()
            if splits[-1] != 'O':
                label_set.add(splits[-1])
    return label_set


class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, type_=None):
        """Reads a tab separated value file."""
        return readfile(input_file, type_)

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = '\t\t'.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir, type_='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir, type_='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir, type_='test'), "test")

    def get_predict_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir, type_='predict'), "predict")


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    ori_sents = []

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split('\t\t')
        ori_sents.append(textlist)
        labellist = example.label

        tokens, labels, valid, label_mask = tokenize_and_preserve_labels(textlist, labellist, tokenizer)

        if len(tokens) >= max_seq_length - 1:
            tokens, labels, valid, label_mask = truncate_sequences(tokens, labels, valid, label_mask, max_seq_length)

        input_ids, input_mask, segment_ids, label_ids, valid, label_mask = build_bert_inputs(tokens, labels, label_map,
                                                                                             max_seq_length, tokenizer,
                                                                                             valid, label_mask)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          seq_len=[len(textlist)]))
    return features, ori_sents


def tokenize_and_preserve_labels(textlist, labellist, tokenizer):
    """Tokenize word and preserve labels."""
    tokens = []
    labels = []
    valid = []
    label_mask = []

    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
                valid.append(1)
                label_mask.append(1)
            else:
                valid.append(0)

    return tokens, labels, valid, label_mask


def truncate_sequences(tokens, labels, valid, label_mask, max_seq_length):
    """Truncate the sequences to the maximum length."""
    tokens = tokens[0:(max_seq_length - 2)]
    labels = labels[0:(max_seq_length - 2)]
    valid = valid[0:(max_seq_length - 2)]
    label_mask = label_mask[0:(max_seq_length - 2)]

    return tokens, labels, valid, label_mask


def build_bert_inputs(tokens, labels, label_map, max_seq_length, tokenizer, valid, label_mask):
    def append_special_token(token, label):
        ntokens.append(token)
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map[label])

    ntokens, segment_ids, label_ids = [], [], []

    append_special_token("[CLS]", "[CLS]")

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_id = label_map[labels[i]] if i < len(labels) else label_map["O"]
        label_ids.append(label_id)

    append_special_token("[SEP]", "[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    padded_sequences = pad_sequences([input_ids, input_mask, segment_ids, label_ids, valid, label_mask], max_seq_length)

    return padded_sequences


def pad_sequences(sequences, max_seq_length):
    """将序列填充到最大长度"""
    padded_sequences = []
    for sequence in sequences:
        while len(sequence) < max_seq_length:
            sequence.append(0)
        padded_sequences.append(sequence)
    return padded_sequences



def get_label_list(train_data_dir, test_data_dir, dev_data_dir):
    label_set = readfile_label(train_data_dir, test_data_dir, dev_data_dir)
    label_list = []
    label_list.append('O')
    label_list.extend(list(label_set))
    label_list.append("[CLS]")
    label_list.append("[SEP]")
    return label_list


def create_tensors(features):
    """创建并返回给定特征属性的张量"""
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_seq_lens = torch.tensor([f.seq_len for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids, all_seq_lens


train_data_dir = "../conll2003/train.txt"
dev_data_dir = "../conll2003/valid.txt"
test_data_dir = "../conll2003/test.txt"
label_list = get_label_list(train_data_dir, test_data_dir, dev_data_dir)
num_labels = len(label_list) + 1
bert_model = 'bert-base-uncased'
# bert_model = 'xlnet-large-cased'
tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=False)
d_model = 768
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0.1
windows_list = [1, 3, 5, 7]
use_bilstm = True

model = LBNER(bert_model=bert_model, d_model=d_model, num_labels=num_labels,
              device=device, dropout=0.1, windows_list=windows_list, use_bilstm=use_bilstm).to(device)

max_seq_length = 128
batch_size = 10

ner_processor = NerProcessor()

train_examples = ner_processor.get_train_examples(train_data_dir)
train_features, _ = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids, all_seq_lens = create_tensors(
    train_features)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids,
                           all_seq_lens)

eval_examples = ner_processor.get_dev_examples(dev_data_dir)
eval_features, _ = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids, all_seq_lens = create_tensors(
    eval_features)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids,
                          all_seq_lens)

EPOCH = 20

train_dataloader = DataLoader(train_data, batch_size=batch_size)
eval_dataloader = DataLoader(eval_data, batch_size=batch_size)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
lr = 5e-5
adam_epsilon = 1e-8
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
max_grad_norm = 1.0

criterion = nn.CrossEntropyLoss(ignore_index=0)


def get_loss(criterion, labels, attention_mask_label, logits, num_labels):
    if attention_mask_label is not None:
        active_loss = attention_mask_label.view(-1) == 1
        active_logits = logits.view(-1, num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)
    else:
        loss = criterion(logits.view(-1, num_labels), labels.view(-1))
    return loss


def train(model, data_loader, optimizer, criterion, max_grad_norm, num_labels, epoch_):
    logging.info('TRAINING')
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(tqdm(data_loader, desc="TRAINING")):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, seq_len = batch

        logits = model(input_ids, segment_ids, input_mask, valid_ids)
        loss = get_loss(criterion, label_ids, l_mask, logits, num_labels)
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(data_loader)

    logging.info(f'\tTrain Loss: {train_loss:.3f}')

def evaluate(model, data_loader, optimizer, criterion, max_grad_norm, num_labels, label_list, epoch_):
    total_loss = 0.0
    logging.info("EVALUATING")
    f1_list = []
    model.eval()
    y_true = []
    y_pred = []

    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for step, batch in enumerate(tqdm(data_loader, desc="Evaluation")):
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, seq_len = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)
        seq_len = seq_len.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids)
            loss = get_loss(criterion, label_ids, l_mask, logits, num_labels)
            total_loss += loss.item()
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()


        for i, label in enumerate(label_ids):
            print()
            temp_1 = []
            temp_2 = []

            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)

                    y_pred.append(temp_2)
                    break
                else:

                    temp_1.append(label_map[label_ids[i][j]])
                    try:
                        temp_2.append(label_map[logits[i][j]])
                    except:
                        temp_2.append('O')
    print('y_true', y_true)
    print('y_pred', y_pred)
    mlb = MultiLabelBinarizer()
    y_true_binary = mlb.fit_transform(y_true)
    y_pred_binary = mlb.transform(y_pred)

    eval_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, average='weighted')
    recall = recall_score(y_true_binary, y_pred_binary, average='weighted')
    f1 = f1_score(y_true_binary, y_pred_binary, average='weighted')
    print(
        f"Epoch {epoch_}, Validation loss: {eval_loss:.2f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    report = classification_report(y_true, y_pred, digits=4)
    logger.info("\n******evaluate on the dev data*******")
    logger.info("\n%s", report)
    print("report")
    print(report)
    temp = report.split('\n')[-3]
    f_eval = eval(temp.split()[-2])
    f1_list.append(f_eval)

    logging.info(f'\t Eval Loss: {eval_loss:.3f}')
    logging.info(f'\t accuracy: {accuracy:.3f}')
    logging.info(f'\t precision: {precision:.3f}')
    logging.info(f'\t recall: {recall:.3f}')
    logging.info(f'\t f1: {f1:.3f}')

    logging.info("eval report")
    logging.info(report)

    output_eval_file = "eval_results.txt"
    with open(output_eval_file, "a") as writer:
        writer.write('*******************epoch*******' + str(epoch_) + '\n')
        writer.write(f'\t Eval Loss: {eval_loss:.3f}' + '\n')
        writer.write(f'\t accuracy: {accuracy:.3f}' + '\n')
        writer.write(f'\t precision: {precision:.3f}' + '\n')
        writer.write(f'\t recall: {recall:.3f}' + '\n')
        writer.write(f'\t f1: {f1:.3f}' + '\n')
        writer.write(report + '\n')


if __name__ == "__main__":
    for epoch_ in trange(EPOCH, desc="Epoch"):
        logging.info(f'Epoch: {epoch_ + 1:02}')
        train(model, train_dataloader, optimizer, criterion, max_grad_norm, num_labels, epoch_)
        evaluate(model, eval_dataloader, optimizer, criterion, max_grad_norm, num_labels, label_list, epoch_)
