import torch
from torch import nn, optim
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


# Load dataset
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
    processed_sentences = [torch.tensor([word_vocab[word] for word in sentence], dtype=torch.long) for sentence in
                           sentences]
    processed_labels = [torch.tensor([label_vocab[label] for label in label], dtype=torch.long) for label in labels]

    return list(zip(processed_sentences, processed_labels))


def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_list = pad_sequence(text_list, padding_value=word_vocab['<pad>'])
    label_list = pad_sequence(label_list, padding_value=label_vocab['<pad>'])
    return text_list, label_list


# the LSTM model
class LSTMTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_dim * 2, output_dim) if bidirectional else nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        outputs, (hidden, _) = self.lstm(embedded)
        predictions = self.fc(outputs)
        tag_scores = torch.log_softmax(predictions, dim=2)
        return tag_scores
        # return predictions


def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

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

            all_predictions.extend(filtered_predictions.tolist())
            all_labels.extend(filtered_labels.tolist())

    eval_loss = total_loss / len(iterator)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return eval_loss, accuracy, precision, recall, f1


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0

    for text, labels in train_loader:
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)
    return train_loss


def train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device):
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        eval_loss, accuracy, precision, recall, f1 = evaluate(model, valid_loader, criterion)

        logging.info(f'Epoch: {epoch + 1:02}')
        logging.info(f'\tTrain Loss: {train_loss:.3f}')
        logging.info(f'\t Eval Loss: {eval_loss:.3f}')
        logging.info(f'\t accuracy: {accuracy:.3f}')
        logging.info(f'\t precision: {precision:.3f}')
        logging.info(f'\t recall: {recall:.3f}')
        logging.info(f'\t f1: {f1:.3f}')


if __name__ == "__main__":
    # load dataset
    sentences, labels = load_data("../conll2003/train.txt")
    word_vocab = build_vocab(sentences)
    label_vocab = build_vocab(labels)
    processed_data = process_data(sentences, labels, word_vocab, label_vocab)
    train_dataset = to_map_style_dataset(processed_data)
    valid_sentences, valid_labels = load_data("../conll2003/valid.txt")
    processed_valid_data = process_data(valid_sentences, valid_labels, word_vocab, label_vocab)
    valid_dataset = to_map_style_dataset(processed_valid_data)

    # define hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    INPUT_DIM = len(word_vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = len(label_vocab)
    NUM_LAYERS = 4
    DROPOUT_RATE = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'BATCH_SIZE {BATCH_SIZE}, HIDDEN_DIM {HIDDEN_DIM}, NUM_LAYERS {NUM_LAYERS}')
    logging.info(f'LEARNING_RATE {LEARNING_RATE}, DROPOUT_RATE {DROPOUT_RATE}')
    logging.info(f'INPUT_DIM {INPUT_DIM}, OUTPUT_DIM {OUTPUT_DIM}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # DEFINING MODEL
    model = LSTMTagger(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE, bidirectional=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=label_vocab['<pad>'])

    N_EPOCHS = 50
    train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion, N_EPOCHS, device)
