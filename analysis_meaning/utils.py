#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions.
"""
from string import Template
from typing import List, Union

import numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForMaskedLM,
                          BertTokenizer, BertModel,
                          # RobertaTokenizer, RobertaModel,
                          GPT2Tokenizer, GPT2Model)

# left and right contexts around the NNC
NNC_TEMPLATE = Template('This is the $nnc,')
# position of the second noun in the template, starting from 0
NNC_POSITION = 4
# tab in the SAP xlsx file
SAP_TYPE = 'classic'
# "neutral" verb for mvrr causal baseline
MVRR_VERB = 'given'
# "neutral" verb for nps causal baseline
NPS_VERB = 'said'
# metric to be used as a distance (cosine, manhattan or euclidean)
METRIC = 'cosine'

sap_dict = {
    'classic': 'NPSNPZMVRR',
    'relative': 'ORCSRC (Staub, 2010)',
    'attachment': 'MultiHighLow attachment (Dillon',
    'agreement': 'agreement subset'
}


def tokenize(sentence: str) -> List[str]:
    """Add space before punctuation and split by space."""
    sent = sentence.replace(',', ' ,').replace('.', ' .').replace('\'', ' \'')
    return sent.split(' ')


def load_tokenizer_and_model(model: str):
    """Load a model and its tokenizer from Hugging face."""
    if model == 'bert':
        # as per https://huggingface.co/bert-base-uncased
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")

    elif model == 'roberta':
        # as per https://huggingface.co/roberta-base
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # model = RobertaModel.from_pretrained('roberta-base')

    elif model == 'gpt2':
        # as per https://huggingface.co/gpt2
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')

    elif model == 'opt':
        # as per https://huggingface.co/facebook/opt-125m
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    else:
        raise NotImplementedError

    print((f'Loaded model with {model.config.hidden_size} features '
           f'and {model.config.num_hidden_layers} layers.'))
    return tokenizer, model


def detect_added_positions(shorter_sent: List[str], longer_sent: List[str]):
    """Return positions that differ in long wrt short sentence."""
    added = []
    enumerated = list(enumerate(longer_sent))
    for position, token in enumerate(shorter_sent):
        while token != enumerated[position][1]:
            added.append(enumerated[position][0])
            enumerated.pop(position)
    return added


def adjust_idx_rand(idxs: Union[str, int, float]) -> List[int]:
    """Parse positions in dataframe, return first and last indexed by 0."""
    if isinstance(idxs, (float, int)):
        return [None, int(idxs) - 1]
    begin, end = idxs.split('-')
    return [int(begin) - 1, int(end) - 1]


def add_emtpy_row_column(chart: np.array) -> np.array:
    """Add an empty row and column to the right of an array."""
    empty_row = np.nan * np.ones((1, chart.shape[1]))
    chart_ext = np.vstack([chart, empty_row])
    empty_col = np.nan * np.ones((chart.shape[1] + 1, 1))
    chart_ext = np.hstack([chart_ext, empty_col])
    return chart_ext


def crop_chart(chart: np.array, begin: int, n_future: int = 10):
    """Crop the beginning and the end of a chart around a position."""
    # look at up n_future tokens into the future
    chart = chart[begin: begin + n_future, begin: begin + n_future]
    return np.expand_dims(chart, 0)
