import json
import numpy
import operator
import pandas
from transformers import BertModel, BertTokenizer
import torch

# Grab models from torch and transformers
TOKENIZER, MODEL, WEIGHTS = (
    BertTokenizer, BertModel, 'bert-base-uncased')

# Load in models and tokenizer
BERT_tokenizer = TOKENIZER.from_pretrained(WEIGHTS)
BERT_model = MODEL.from_pretrained(WEIGHTS)


def data_load():
    '''
    Function helps us load data from the json file into an array 
    '''
    file_path = 'Data/yummly.json'

    # Open the json file
    j_f = open(file_path)
    # Load the file
    data = json.load(j_f)

    df = pandas.DataFrame(data)

    return df


def data_process(data_frame):
    '''
    Data is formatted for BERT model, padded, and masked
    '''

    # Join ingredients list into string separated by commas
    data_frame['ingredients'] = data_frame['ingredients'].apply(', '.join)

    # Process data through tokenizer for BERT model
    processed_data = data_frame['ingredients'].apply(
        (lambda x: BERT_tokenizer.encode(x, add_special_tokens=True)))
    # Find maximum padding
    max_counter = 0
    for ing in processed_data.values:
        length = len(ing)
        if length > max_counter:
            max_counter = length
    # Pad data in ingredients lists
    padded_data = numpy.array([ing + [0]*(max_counter - len(ing))
                               for ing in processed_data.values])
    # Mask padded_data
    key = numpy.where(padded_data != 0, 1, 0)

    return padded_data, key


def model_run(padded_data, masked_data):
    # Run data gathered from processing into torch
    data_mask = torch.tensor(masked_data)
    data_input = torch.tensor(padded_data)

    # Run through model
    with torch.no_grad():
        embeddings = BERT_model(data_input, attention_mask=data_mask)

    features = embeddings[0][:, 0, :].numpy()

    return features
