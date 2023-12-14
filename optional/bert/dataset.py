import torch
from torch.utils import data

data_dir = "../../conll2003/"

class Input():
    def __init__(self, guid, text, label=None, segment_ids=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.segment_ids = segment_ids
        
def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
    
    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data

class CoNLL2003Processor():
    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            label = label
            examples.append(Input(guid=guid, text=text_a, label=label))
        return examples
    def get_train(self):
        return self._create_examples(readfile(data_dir + "train.txt"), "train")
    def get_valid(self):
        return self._create_examples(readfile(data_dir + "valid.txt"), "valid")
    def get_test(self):
        return self._create_examples(readfile(data_dir + "test.txt"), "test")
    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                "[CLS]", "[SEP]", "X"]
    
class CoNLL2003DataSet(data.Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        label = input_example.label
        word_tokens = ['[CLS]']
        label_list = ['[CLS]']
        label_mask = [0]  
        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['[CLS]']]

        for word, label in zip(text.split(), label):
            tokenized_word = self.tokenizer.tokenize(word)

            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)
            for i in range(1, len(tokenized_word)):
                label_list.append('X')
                label_ids.append(self.label_map['X'])
                label_mask.append(0)

        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        word_tokens.append('[SEP]')
        label_list.append('[SEP]')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['[SEP]'])
        label_mask.append(0)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            label_ids.append(self.label_map['X'])
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)
    