# using LSTM and BERT
from transformers import (PreTrainedModel, AutoModel, AutoConfig)
import math
import random
import numpy as np
from transformers import AutoModelWithLMHead
import torch
from torch import nn, optim
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    """
      a sentence is the first word of each line before seeing a blank line
      same for label but is the forth word
    """
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    sentences, labels = [], []
    sentence, sentence_labels = [], []
    for line in lines:
        if line.startswith("-DOCSTART-") or line == "\n":
            if sentence:
                sentences.append(sentence)
                labels.append(sentence_labels)
                sentence, sentence_labels = [], []
        else:
            parts = line.split()
            sentence.append(parts[0])
            sentence_labels.append(parts[-1])

    return sentences, labels

def build_vocab(data_iter):
    specials = ['<unk>', '<pad>']
    vocab = build_vocab_from_iterator(data_iter, specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def process_data(sentences, labels, word_vocab, label_vocab):
    processed_sentences = [torch.tensor([word_vocab[word] for word in sentence], dtype=torch.long) for sentence in sentences]
    processed_labels = [torch.tensor([label_vocab[label] for label in label], dtype=torch.long) for label in labels]

    return list(zip(processed_sentences, processed_labels))

def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_list = pad_sequence(text_list, padding_value=word_vocab['<pad>'])
    label_list = pad_sequence(label_list, padding_value=label_vocab['<pad>'])
    return text_list, label_list



# load dataset
sentences, labels = load_data("../conll2003/train.txt")
word_vocab = build_vocab(sentences)
label_vocab = build_vocab(labels)
processed_data = process_data(sentences, labels, word_vocab, label_vocab)
train_dataset = to_map_style_dataset(processed_data)
valid_sentences, valid_labels = load_data("../conll2003/valid.txt")
processed_valid_data = process_data(valid_sentences, valid_labels, word_vocab, label_vocab)
valid_dataset = to_map_style_dataset(processed_valid_data)


# class BiLSTM(nn.Module):
#     def __init__(self, hidden_size):
#         super(BiLSTM, self).__init__()
#         # self.setup_seed(seed)
#         self.forward_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=False, batch_first=True)
#         self.backward_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=False, batch_first=True)
#
#     def forward(self, x):
#         batch_size, max_len, feat_dim = x.shape
#         out1, _ = self.forward_lstm(x)
#         reverse_x = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
#         for i in range(max_len):
#             reverse_x[:, i, :] = x[:, max_len - 1 - i, :]
#
#         out2, _ = self.backward_lstm(reverse_x)
#
#         output = torch.cat((out1, out2), 2)
#         return output, (1, 1)
#
class BiLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTM, self).__init__()
        # Create a bidirectional LSTM
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        # Pass the input through the bidirectional LSTM
        output, _ = self.bilstm(x)
        return output, (1, 1)

# bert dropout use_bilstm windows_list d_model num_labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LBNER(nn.Module):
    def __init__(self, bert_model, d_model, num_labels, device, dropout=0.1, windows_list=None, use_bilstm=True):
        super(LBNER, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.use_bilstm = use_bilstm
        self.windows_list = windows_list
        self.d_model = d_model
        self.num_labels = num_labels
        self.linear = nn.Linear(self.d_model, self.num_labels)
        self.device = device

        if self.windows_list != None:
            if self.use_bilstm:
                self.bilstm_s = nn.ModuleList([BiLSTM(self.d_model) for _ in self.windows_list])
            else:
                self.bilstm_layers = nn.ModuleList(
                    [nn.LSTM(self.d_model, self.d_model, num_layers=1, bidirectional=False, batch_first=True) for _ in
                     self.windows_list])


    def windows_sequence(self, sequence_output, windows, lstm_layer):
        batch_size, max_len, feat_dim = sequence_output.shape
        local_final = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device=self.device)
        for i in range(max_len):
            index_list = []
            for u in range(1, windows // 2 + 1):
                if i - u >= 0:
                    index_list.append(i - u)
                if i + u <= max_len - 1:
                    index_list.append(i + u)
            index_list.append(i)
            index_list.sort()
            temp = sequence_output[:, index_list, :]
            out, (h, b) = lstm_layer(temp)
            local_f = out[:, -1, :]
            local_final[:, i, :] = local_f
        return local_final

    # bert dropout use_bilstm windows_list d_model num_labels

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
    #             attention_mask_label=None):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = \
        self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        # valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)

        # for i in range(batch_size):
        #     jj = -1
        #     for j in range(max_len):
        #         if valid_ids[i][j].item() == 1:
        #             jj += 1
        #             valid_output[i][jj] = sequence_output[i][j]
        valid_output = torch.stack([sequence_output[i][valid_ids[i].bool()] for i in range(batch_size)])

        sequence_output = self.dropout(valid_output)

        # mutiple_windows = []

        mutiple_windows = [
            self.windows_sequence(sequence_output, window, self.bilstm_layers[i])
            for i, window in enumerate(self.windows_list)
            if self.use_bilstm
        ]

        # for i, window in enumerate(self.windows_list):
        #     if self.use_bilstm:
        #         local_final = self.windows_sequence(sequence_output, window, self.bilstm_layers[i])
        #     mutiple_windows.append(local_final)

        muti_local_features = torch.stack(mutiple_windows, dim=2)
        sequence_output = sequence_output.unsqueeze(dim=2)
        d_k = sequence_output.size(-1)
        attn = torch.matmul(sequence_output, muti_local_features.permute(0, 1, 3, 2)) / math.sqrt(d_k)
        attn = torch.softmax(attn, dim=-1)
        local_features = torch.matmul(attn, muti_local_features).squeeze()
        sequence_output = sequence_output.squeeze()
        sequence_output = sequence_output + local_features

        logits = self.linear(sequence_output)
        return logits

        # if labels is not None:
        #
        #     loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        #     # Only keep active parts of the loss
        #     # attention_mask_label = None
        #     if attention_mask_label is not None:
        #         active_loss = attention_mask_label.view(-1) == 1
        #         active_logits = logits.view(-1, self.num_labels)[active_loss]
        #         active_labels = labels.view(-1)[active_loss]
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     return loss
        # else:
        #
        #     return logits


