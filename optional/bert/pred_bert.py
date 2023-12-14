import torch
import torch.nn.functional as F
from seqeval.metrics import accuracy_score, f1_score, classification_report
from torch.utils import data
from tqdm import trange, tqdm
from transformers import BertTokenizer, AdamW

from dataset import CoNLL2003DataSet, CoNLL2003Processor
from model import BertTagger
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CoNLL2003Processor()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
label_list = processor.get_labels()
tags_vals = label_list
label_map = {label: i for i, label in enumerate(label_list)}
train_data = processor.get_train()
val_data = processor.get_valid()
test_data = processor.get_test()
training_set = CoNLL2003DataSet(train_data, tokenizer, label_map, max_len=128)
eval_set = CoNLL2003DataSet(val_data, tokenizer, label_map, max_len=256)
test_set = CoNLL2003DataSet(test_data, tokenizer, label_map, max_len=256)

train_iter = data.DataLoader(dataset=training_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=12)
eval_iter = data.DataLoader(dataset=eval_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12)
test_iter = data.DataLoader(dataset=test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12)

model = BertTagger.from_pretrained('bert-base-cased', num_labels=len(label_map))
model.load_state_dict(torch.load('models/bert.pt'))
model.to(device)
nb_eval_steps = 0
predictions, true_labels = [], []
input_ids = []

for batch in tqdm(test_iter):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
    input_ids.extend(b_input_ids)
    with torch.no_grad():
        tmp_eval_loss, logits, reduced_labels = model(b_input_ids,
                                                        token_type_ids=b_token_type_ids,
                                                        attention_mask=b_input_mask,
                                                        labels=b_labels,
                                                        label_masks=b_label_masks)
    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    reduced_labels = reduced_labels.to('cpu').numpy()
    labels_to_append = []
    predictions_to_append = []
    for prediction, r_label in zip(logits, reduced_labels):
        preds = []
        labels = []
        for pred, lab in zip(prediction, r_label):
            if lab.item() == -1:  # masked label; -1 means do not collect this label
                continue
            preds.append(pred)
            labels.append(lab)
        predictions_to_append.append(preds)
        labels_to_append.append(labels)
    predictions.extend(predictions_to_append)
    true_labels.extend(labels_to_append)


import numpy as np
def decode_input(l):
    res = []
    for line in l:
        if 0 in line:
            line = line[:np.argmin(line)]
        decoded = tokenizer.convert_ids_to_tokens(line)
        res.append(decoded)
    return res

def decode(l):
    res = []
    for line in l:
        res.append([tags_vals[i] for i in line])
    return res    

pred_tags = decode(predictions)
test_tags = decode(true_labels)
input_ids = torch.stack(input_ids).cpu().numpy()
input_sents = decode_input(input_ids)

def write_bert():
    p_1d = [item for sublist in pred_tags for item in sublist]
    t_1d = [item for sublist in test_tags for item in sublist]
    idx = 0
    with open("../../conll2003/test.txt","r") as f1:
        with open("out/3036197122.bert.test.txt","w") as f2:
            for line in f1:
                if line!="\n" and line!="":
                    words = line.strip().split()
                    if words[0].startswith("-DOCSTART-"):
                        f2.write(line)
                        continue
                    if t_1d[idx] == words[-1]:
                        words[-1] = p_1d[idx]+"\n"
                        idx+=1
                    else:
                        words[-1] = "*******\n"
                        idx+=1

                    f2.write(" ".join(words))
                else:
                    f2.write("\n")

def get_all_input_sents():
    with open("../../conll2003/test.txt","r") as f:
        sents = []
        sent = []
        for line in f:
            if line.startswith("-DOCSTART-"):
                continue
            if line=="\n":
                if sent == []:
                    continue
                sents.append(sent)
                sent = []
            else:
                sent.append(line.strip().split()[0])
    return sents
sents = get_all_input_sents()
for sent in sents:
    if len(sent)<=2:
        print(sent)
        break

with open("logs/compare.txt","w") as f:
    for i in range(len(sents)):
        f.write(" ".join(str(input_ids[i]))+"\n")
        f.write(" ".join(input_sents[i])+"\n")
        f.write(" ".join(sents[i])+"\n")
        f.write(" ".join(pred_tags[i])+"\n")
        f.write(" ".join(test_tags[i])+"\n")

        f.write("\n")
        f.write("**************************\n")

if __name__ == "__main__":
    print("*****************starting to write to file*****************")
    write_bert()