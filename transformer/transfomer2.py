import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.optim as optim

def load_data(file_path):
    sentences, tags = [], []
    with open(file_path, encoding='utf-8') as file:
        sentence, tag = [], []
        for line in file:
            if line == '\n':
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence, tag = [], []
            else:
                word, _, _, label = line.strip().split()
                sentence.append(word)
                tag.append(label)
        if sentence:
            sentences.append(sentence)
            tags.append(tag)
    return sentences, tags

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_vocab, tag_vocab):
        self.sentences = sentences
        self.tags = tags
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (
            torch.tensor([self.word_vocab.get_stoi()[word] for word in self.sentences[idx]], dtype=torch.long),
            torch.tensor([self.tag_vocab.get_stoi()[tag] for tag in self.tags[idx]], dtype=torch.long)
        )

def collate_batch(batch):
    word_list, tag_list = zip(*batch)
    word_list = pad_sequence(word_list, batch_first=True, padding_value=word_vocab['<pad>'])
    tag_list = pad_sequence(tag_list, batch_first=True, padding_value=tag_vocab['<pad>'])
    return word_list, tag_list

tokenizer = get_tokenizer('basic_english')

# Load data
train_sentences, train_tags = load_data('../conll2003/train.txt') # Replace with your train dataset path
valid_sentences, valid_tags = load_data('../conll2003/valid.txt') # Replace with your valid dataset path

# Build vocabularies
def build_vocab(data):
    counter = Counter()
    for sentence in data:
        counter.update(sentence)
    return build_vocab_from_iterator([counter], specials=['<unk>', '<pad>'])

word_vocab = build_vocab(train_sentences + valid_sentences)
tag_vocab = build_vocab(train_tags + valid_tags)

# Create datasets and data loaders
train_dataset = NERDataset(train_sentences, train_tags, word_vocab, tag_vocab)
valid_dataset = NERDataset(valid_sentences, valid_tags, word_vocab, tag_vocab)

BTACH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BTACH_SIZE, collate_fn=collate_batch, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BTACH_SIZE, collate_fn=collate_batch)

import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, ntag: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(d_model, ntag)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for data, targets in progress_bar:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# def train(model, train_loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for batch, (data, targets) in enumerate(train_loader):
#         data, targets = data.to(device), targets.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(train_loader)

def evaluate(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, (data, targets) in enumerate(valid_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(valid_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(word_vocab.get_stoi())  # 词汇表的大小
d_model = 512  # 嵌入维度
nhead = 8  # 多头注意力机制中的头数
d_hid = 2048  # 前馈网络的维度
nlayers = 6  # 编码器层的数量
dropout = 0.1  # Dropout层的丢弃率
ntag = len(tag_vocab.get_stoi())

model = TransformerModel(ntokens, ntag, d_model, nhead, d_hid, nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab.get_stoi()['<pad>'])
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(1, 11):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, valid_loader, criterion, device)
    print(f"Epoch {epoch}, Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}")