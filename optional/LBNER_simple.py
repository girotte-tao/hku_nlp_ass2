import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoModel, AutoTokenizer

class CoNLL2003Dataset(Dataset):
    def __init__(self, file_path, tokenizer, word_vocab, label_vocab):
        self.data = self.read_file(file_path)
        self.tokenizer = tokenizer
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

    def read_file(self, file_path):
        file_data = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == "" or line.startswith("-DOCSTART-"):
                    if tokens:
                        file_data.append((tokens, labels))
                        tokens, labels = [], []
                else:
                    word, _, _, tag = line.split()
                    tokens.append(word)
                    labels.append(tag)
        return file_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        token_ids = self.tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, max_length=128, return_tensors="pt").input_ids.squeeze(0)
        label_ids = [self.label_vocab[label] for label in labels]
        label_ids += [self.label_vocab['O']] * (128 - len(label_ids))  # Padding labels
        return token_ids, torch.tensor(label_ids, dtype=torch.long)

    @staticmethod
    def yield_tokens(data_iter):
        for tokens, _ in data_iter:
            yield tokens



bert_model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def build_vocab(file_path):
    dataset = CoNLL2003Dataset(file_path, tokenizer, None, None)
    vocab = build_vocab_from_iterator(dataset.yield_tokens(dataset.data), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

word_vocab = build_vocab('../conll2003/train.txt')
label_vocab = build_vocab('../conll2003/va.txt')  # assuming labels are also in this file
