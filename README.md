# NER Model Training and Prediction Guide

This README provides instructions for training and predicting Named Entity Recognition (NER) using LSTM, Transformer, LBNER, and BertTagger models.

Github: https://github.com/girotte-tao/hku_nlp_ass2

## Environment

The project uses conda to build the environment.
Please use the environment.yml file to build the environment.

## Training Models

To train the models, follow the instructions specific to each model:

### LSTM and Transformer

1. Navigate to the respective directory containing the LSTM or Transformer model file.
2. Run the Python file named after the model to start training:
   ```bash
   python lstm_model.py    # For LSTM
   python transformer_model.py  # For Transformer

### LBNER

1. Navigate to the optional/lbner directory.
2. Run the run_LBNER.py file to train the LBNER model:
   ```bash
   python run_LBNER.py

### BertTagger

1. Navigate to the optional/bert directory.
2. Run the train.py file to train the BertTagger model:
   ```bash
   python train.py

## Prediction

### LSTM, Transformer and LBNER

To predict with the LSTM, Transformer, and LBNER models, modify the if __name__ == "__main__" section in their respective files by changing do_train to do_predict:

if `__name__` == `"__main__"`:
    
    do_predict == True
    do_train == False

Please do training before do prediction because model files will be loaded when do prediction.

### BertTagger

For the Bert model, run the following command for prediction:


## Log

To see logs, please turn to the log directory of the respective model directories.
For log of BERT, the last log of the report in the train log is the report on the test dataset.