import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import torch.optim as optim
from seqeval.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from torchtext.data.functional import to_map_style_dataset
import logging
import os
from datetime import datetime
from tqdm import tqdm
import optuna
import torch.nn as nn
import math
import matplotlib.pyplot as plt
torch.manual_seed(98)

#  ----- log config -----
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
            torch.tensor([self.word_vocab[word] for word in self.sentences[idx]], dtype=torch.long),
            torch.tensor([self.tag_vocab[tag] for tag in self.tags[idx]], dtype=torch.long)
        )


def process_data(sentences, labels, word_vocab, label_vocab):
    processed_sentences = [torch.tensor([word_vocab[word] for word in sentence], dtype=torch.long) for sentence in
                           sentences]
    processed_labels = [torch.tensor([label_vocab[label] for label in label], dtype=torch.long) for label in labels]

    return list(zip(processed_sentences, processed_labels))


def build_vocab(data_iter):
    specials = ['<unk>', '<pad>']
    vocab = build_vocab_from_iterator(data_iter, specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab


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

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


def train(model, train_loader, optimizer, criterion, device):
    logging.info('TRAINING...')
    model.train()
    total_loss = 0.0
    total = len(train_loader)
    progress_bar = tqdm(train_loader, desc='Training', leave=True)

    for data, targets in progress_bar:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / total


def evaluate(model, valid_loader, criterion, device, tag_vocab):
    logging.info('EVALUATING...')
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc='Validating', leave=True)
        total = len(valid_loader)
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            output = output.view(-1, output.shape[-1])
            targets = targets.view(-1)

            loss = criterion(output, targets)
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            mask = targets != tag_vocab['<pad>']
            filtered_predictions = predicted[mask]
            filtered_targets = targets[mask]

            all_predictions.append([tag_vocab.lookup_token(index) for index in filtered_predictions.tolist()])
            all_targets.append([tag_vocab.lookup_token(index) for index in filtered_targets.tolist()])

    eval_loss = total_loss / total
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    report = classification_report(all_targets, all_predictions, digits=4)
    return eval_loss, accuracy, precision, recall, f1, report


def train_and_eval(epoch, model, optimizer, criterion, device, train_loader, valid_loader, tag_vocab):
    val_loss, accuracy, precision, recall, f1 = -1, -1, -1, -1, -1
    train_losses = []
    eval_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores =[]
    for epoch in range(1, epoch + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, accuracy, precision, recall, f1, report = evaluate(model, valid_loader, criterion, device, tag_vocab)
        train_losses.append(train_loss)
        eval_losses.append(val_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        print(
            f"Epoch {epoch}, Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        logging.info(f'Epoch: {epoch:02}')
        logging.info(f'\tTrain Loss: {train_loss:.3f}')
        logging.info(f'\t Eval Loss: {val_loss:.3f}')
        logging.info(f'\t accuracy: {accuracy:.3f}')
        logging.info(f'\t precision: {precision:.3f}')
        logging.info(f'\t recall: {recall:.3f}')
        logging.info(f'\t f1: {f1:.3f}')
        logging.info('\n' + report)

    plt.figure(figsize=(15, 10))

    # 绘制训练损失
    plt.subplot(3, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制评估损失
    plt.subplot(3, 2, 2)
    plt.plot(eval_losses, label='Eval Loss')
    plt.title('Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制准确率
    plt.subplot(3, 2, 3)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # 绘制精确度
    plt.subplot(3, 2, 4)
    plt.plot(precisions, label='Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    # 绘制召回率
    plt.subplot(3, 2, 5)
    plt.plot(recalls, label='Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    # 绘制F1得分
    plt.subplot(3, 2, 6)
    plt.plot(f1_scores, label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    return f1


def objective(trial):
    train_sentences, train_tags = load_data('../conll2003/train.txt')
    valid_sentences, valid_tags = load_data('../conll2003/valid.txt')
    word_vocab = build_vocab(train_sentences)
    tag_vocab = build_vocab(train_tags)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(word_vocab)
    ntag = len(tag_vocab)
    LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    d_model = trial.suggest_categorical("d_model", [256, 512, 1024])
    nhead = trial.suggest_categorical("nhead", [4, 8, 16])
    d_hid = trial.suggest_categorical("d_hid", [1024, 512, 256])
    nlayers = trial.suggest_int("nlayers", 2, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    logging.info(f'device {device}')
    logging.info(f'ntokens {ntokens} ntag{ntag}')
    logging.info(f'd_model{d_model} nhead{nhead} d_hid{d_hid} nlayers{nlayers} dropout{dropout}')

    def collate_batch(batch):
        word_list, tag_list = zip(*batch)
        word_list = pad_sequence(word_list, padding_value=word_vocab['<pad>'])
        tag_list = pad_sequence(tag_list, padding_value=tag_vocab['<pad>'])
        return word_list, tag_list

    train_loader = DataLoader(to_map_style_dataset(process_data(train_sentences, train_tags, word_vocab, tag_vocab)),
                              batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
    valid_loader = DataLoader(to_map_style_dataset(process_data(valid_sentences, valid_tags, word_vocab, tag_vocab)),
                              batch_size=BATCH_SIZE, collate_fn=collate_batch)

    model = TransformerModel(ntokens, ntag, d_model, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab['<pad>'])
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    EPOCH = 50
    f1 = train_and_eval(EPOCH, model, optimizer, criterion, device, train_loader, valid_loader, tag_vocab)
    return f1

def load_data_lines(file_path):
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    return lines

def predict(model, iterator, criterion, device, label_vocab, sentences, lines):
    with open('3036197122.transformer.test1.txt', 'w') as f:
        model.eval()
        total_loss = 0
        all_predictions = []
        all_predictions_ex = []
        all_words_ex = [word for sentence in sentences for word in sentence]
        all_labels = []
        line_cnt = len(lines)

        with torch.no_grad():
            for text, labels in iterator:
                text, labels = text.to(device), labels.to(device)
                predictions = model(text)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                loss = criterion(predictions, labels)
                total_loss += loss.item()

                _, predicted = torch.max(predictions, dim=1)

                non_pad_elements = labels != label_vocab['<pad>']
                filtered_predictions = predicted[non_pad_elements]
                filtered_labels = labels[non_pad_elements]

                predictions = [label_vocab.lookup_token(index) for index in filtered_predictions.tolist()]
                labels = [label_vocab.lookup_token(index) for index in filtered_labels.tolist()]

                all_predictions.append(predictions)
                all_labels.append(labels)

                all_predictions_ex.extend(predictions)

        line_index = 0
        word_index = 0
        while line_index < line_cnt:
            if lines[line_index] == "\n":
                f.write(lines[line_index])
                line_index += 1
            else:
                line_info = lines[line_index].strip().split()

                logging.info(f'{line_info[0]}, {all_words_ex[word_index]}  word_index{word_index}')
                assert line_info[0] == all_words_ex[word_index]
                f.write(f'{line_info[0]} {line_info[1]} {line_info[2]} {all_predictions_ex[word_index]}\n')
                word_index += 1
                line_index += 1
        eval_loss = total_loss / len(iterator)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        report = classification_report(all_labels, all_predictions, digits=4)
        logging.info(report)
        return eval_loss, accuracy, precision, recall, f1, report, all_predictions

def do_predict():
    model = torch.load('model.pth')
    sentences, labels = load_data("../conll2003/train.txt")
    word_vocab = build_vocab(sentences)
    label_vocab = build_vocab(labels)
    test_sentences, test_labels = load_data("../conll2003/test.txt")
    lines = load_data_lines("../conll2003/test.txt")

    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def collate_batch(batch):
        text_list, label_list = zip(*batch)
        text_list = pad_sequence(text_list, padding_value=word_vocab['<pad>'])
        label_list = pad_sequence(label_list, padding_value=label_vocab['<pad>'])
        return text_list, label_list

    test_loader = DataLoader(to_map_style_dataset(process_data(test_sentences, test_labels, word_vocab, label_vocab)),
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    criterion = nn.CrossEntropyLoss(ignore_index=label_vocab['<pad>'])
    predict(model, test_loader,
            criterion,
            device, label_vocab,
            test_sentences, lines)



def print_trial_info(study, trial):
    # 打印每次 trial 完成时的信息
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}.")
    print(f"Best trial so far: Trial {study.best_trial.number}")
    logging.info(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}.")
    logging.info(f"Best trial so far: Trial {study.best_trial.number}")


def main():
    train_sentences, train_tags = load_data('../conll2003/train.txt')
    valid_sentences, valid_tags = load_data('../conll2003/valid.txt')
    word_vocab = build_vocab(train_sentences)
    tag_vocab = build_vocab(train_tags)
    processed_data = process_data(train_sentences, train_tags, word_vocab, tag_vocab)
    train_dataset = to_map_style_dataset(processed_data)

    processed_valid_data = process_data(valid_sentences, valid_tags, word_vocab, tag_vocab)
    valid_dataset = to_map_style_dataset(processed_valid_data)

    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(word_vocab)
    d_model = 512
    nhead = 4
    d_hid = 1024
    nlayers = 3
    dropout = 0.11424641724202106
    ntag = len(tag_vocab)
    LEARNING_RATE = 0.0879668405212036

    def collate_batch(batch):
        word_list, tag_list = zip(*batch)
        word_list = pad_sequence(word_list, padding_value=word_vocab['<pad>'])
        tag_list = pad_sequence(tag_list, padding_value=tag_vocab['<pad>'])
        return word_list, tag_list

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    logging.info(f'device {device}')
    logging.info(f'ntokens {ntokens} ntag{ntag}')
    logging.info(f'd_model{d_model} nhead{nhead} d_hid{d_hid} nlayers{nlayers} dropout{dropout}')

    model = TransformerModel(ntokens, ntag, d_model, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab['<pad>'])
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    EPOCH = 60
    train_and_eval(EPOCH, model, optimizer, criterion, device, train_loader, valid_loader, tag_vocab)
    torch.save(model, 'model.pth')

if __name__ == "__main__":
    do_trail = False
    do_prediction = True
    do_train = False

    if do_prediction:
        do_predict()

    if do_trail:
        logging.info('DO TRIAL...')
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, callbacks=[print_trial_info])

        print(study.best_params)
    if do_train:
        main()
