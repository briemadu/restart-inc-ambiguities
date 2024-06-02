#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates the incremental embeddings of pre-trained models on
the stimuli sentences from three datasets:

- SAP benchmark: the classic garden path sentences in the
first tab of the released xlsx.
- NNC: the noun noun compounds withe a fixed left context defined in TEMPLATE.

See the README for the references and sources.

The constructed tensors (triangular prisms, as defined in the paper) are
saved as h5 data files, with one dataset for each sentence variation.

Sentences that get subtokenized by the pretrained models are ignored, as it
would become harder to work with an object that contains misaligments.

If you are using the original SAP data, 'Unnamed: 5' should be 'Unnamed: 3' in
line 111. We have more columns due to our own annotation.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import h5py
import json
from tqdm import tqdm

from utils import load_tokenizer_and_model
from incstates import IncStatesCreator


parser = argparse.ArgumentParser(
    description='Extract states for the NNC and garden path stimuli.')
parser.add_argument(
    '-out_path',
    default='outputs/embeddings/',
    type=str, help='Path where to save the states h5py files.')
parser.add_argument(
    '-preproc_path',
    default='./outputs/preprocessed_stimuli.csv',
    type=str, help='Path where to save the preprocessed stimuli .csv file.')
parser.add_argument(
    '-model',
    default='bert',
    choices=['bert', 'roberta', 'opt', 'gpt2'],
    type=str, help='Pretrained model.')
args = parser.parse_args()

os.makedirs(Path(args.out_path), exist_ok=True)

df = pd.read_csv(Path(args.preproc_path), index_col=0)
tokenizer, model = load_tokenizer_and_model(args.model)

for source, data in df.groupby('source'):
    skipped = []
    h5file = Path(f'{args.out_path}{args.model}_{source}_embeddings.h5')
    with h5py.File(h5file, 'w') as f:
        for idx, row in tqdm(data.iterrows(), desc=source):

            stimulus = IncStatesCreator(row.stimulus,
                                        args.model,
                                        tokenized=True)
            baseline = IncStatesCreator(row.baseline,
                                        args.model,
                                        tokenized=True)
            try:
                stimulus.build_states(tokenizer, model)
                baseline.build_states(tokenizer, model)
            except AssertionError:
                print(f'  Skipped item {idx} due to subtokenization issues.')
                skipped.append(idx)
                continue

            f.create_dataset(f'{idx}_stimulus', data=stimulus.states)
            f.create_dataset(f'{idx}_baseline', data=baseline.states)

    skipped_path = Path(f'{args.out_path}{args.model}_{source}_skipped.json')
    with open(skipped_path, 'w') as f:
        json.dump(skipped, f)
