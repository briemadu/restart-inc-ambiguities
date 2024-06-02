#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes a distance function with respect to three positions (first, previous
and last) for every state. This is slow, so we save the files as h5py to not
have to repeat it multiple times.
"""

import argparse
from pathlib import Path
from typing import Callable

import h5py
import pandas as pd
from sklearn.metrics.pairwise import paired_distances
from tqdm import tqdm

from incstates import IncStatesAnalyser
from utils import METRIC


parser = argparse.ArgumentParser(
    description='Extract states for the NNC, VN and garden path stimuli.')
parser.add_argument(
    '-out_path',
    default='outputs/triangles/',
    type=str, help='Path where to save the states h5py files.')
parser.add_argument(
    '-states_path',
    default='outputs/embeddings/',
    type=str, help='Path from where to load embeddings.')
parser.add_argument(
    '-preproc_path',
    default='./outputs/preprocessed_stimuli.csv',
    type=str, help='Path where to save the preprocessed stimuli .csv file.')
parser.add_argument(
    '-model',
    default='bert',
    choices=['bert', 'roberta', 'opt', 'gpt2'],
    type=str, help='Pretrained model.')
parser.add_argument(
    '-distance',
    default=METRIC,
    choices=['euclidean', 'manhattan', 'cosine'],
    type=str, help='Which vector distance metric to use.')
args = parser.parse_args()


prefix = f'{args.states_path}/{args.model}'
paths = {
    'nnc': f'{prefix}_nnc_embeddings.h5',
    'classic-nps': f'{prefix}_classic-nps_embeddings.h5',
    'classic-mvrr': f'{prefix}_classic-mvrr_embeddings.h5',
    'classic-nps_for-causal': f'{prefix}_classic-nps_for-causal_embeddings.h5',
    'classic-mvrr_for-causal': f'{prefix}_classic-mvrr_for-causal_embeddings.h5',
}


def dist_func(*x) -> Callable:
    """Return a specific distance function between two vectors."""
    return paired_distances(*x, metric=args.distance)


data_df = pd.read_csv(args.preproc_path, index_col=0)

for source, data in data_df.groupby('source'):

    path = paths[source]
    with h5py.File(Path(path), 'r') as f:
        prisms = {key: value[:] for key, value in f.items()}

    h5file = Path(f'{args.out_path}{args.model}_{source}_{args.distance}.h5')
    with h5py.File(h5file, 'w') as f:

        for idx, row in tqdm(data.iterrows(), desc=source):
            if f'{idx}_stimulus' not in prisms:
                # some sentences were not processed due to subtokenization
                continue

            stimulus_states = prisms[f'{idx}_stimulus']
            obj_stimulus = IncStatesAnalyser(row.stimulus,
                                             states=stimulus_states,
                                             tokenized=True)
            obj_stimulus.compute_distance_charts(dist_func)

            baseline_states = prisms[f'{idx}_baseline']
            obj_baseline = IncStatesAnalyser(row.baseline,
                                             states=baseline_states,
                                             tokenized=True)
            obj_baseline.compute_distance_charts(dist_func)

            for mode in obj_stimulus.modes:
                f.create_dataset(f'{idx}_stimulus_{mode}',
                                 data=obj_stimulus.inc_charts[mode])
                f.create_dataset(f'{idx}_baseline_{mode}',
                                 data=obj_baseline.inc_charts[mode])
