#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads all stimuli from the two sources and saves them into a
standardised file preprocessed_stimuli.csv.

Positions are indexed from 0.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (adjust_idx_rand, sap_dict, tokenize,
                   NNC_TEMPLATE, NNC_POSITION, SAP_TYPE, MVRR_VERB, NPS_VERB)


parser = argparse.ArgumentParser(
    description='Extract states for the NNC, VN and garden path stimuli.')
parser.add_argument(
    '-nnc_path',
    default='./data/noun_compound_senses/dataset/en/neutral/P1_sents.csv',
    type=str, help='Path to NNC data csv file.')
parser.add_argument(
    '-sap_path',
    default='./data/sap-benchmark/Items for all subsets-annotated.xlsx',
    type=str, help='Path to SAP data xlsx file.')
parser.add_argument(
    '-preproc_path',
    default='./outputs/preprocessed_stimuli.csv',
    type=str, help='Path where to save the preprocessed stimuli .csv file.')
args = parser.parse_args()

stimuli = {}


# ################################# NNC data ##################################

data = pd.read_csv(Path(args.nnc_path))

for idx, nnc in tqdm(enumerate(data.compound), desc='nnc stimuli'):

    sentence = tokenize(NNC_TEMPLATE.substitute(nnc=nnc))
    sentence_baseline = tokenize(NNC_TEMPLATE.substitute(nnc=nnc.split()[0]))

    stimuli[len(stimuli)] = {
        'source': 'nnc',
        'stimulus': " ".join(sentence),
        'baseline': " ".join(sentence_baseline),
        'disamb_position_ambiguous': NNC_POSITION,
        'disamb_position_baseline': NNC_POSITION,
        'amb_position_ambiguous': NNC_POSITION - 1,
        'amb_position_baseline': NNC_POSITION - 1,
        'orig_idx': idx
    }

# ################################# SAP data ##################################

xls_data = pd.ExcelFile(Path(args.sap_path))
data = pd.read_excel(xls_data, sap_dict[SAP_TYPE])

if SAP_TYPE == 'classic':
    data.rename(columns={'Unnamed: 5': 'unambiguous'}, inplace=True)

for idx, row in tqdm(data.iterrows(), desc='sap stimuli'):
    # ignore empty rows
    if pd.isna(row.ambiguous) or 'NPZ' in row.condition:
        continue

    condition = row.condition.split('_')[0].lower()
    sentence = tokenize(row.ambiguous)
    baseline_sentence = tokenize(row.unambiguous)

    # -1 because we count from 0
    disamb_position_ambiguous = int(row.disambPositionAmb) - 1
    disam_position_baseline = int(row.disambPositionUnamb) - 1
    # we consider the first verb as the main ambiguous token, but in
    # NP/S the direct object is also ambiguous
    # its position can later be infered from the position of the main verb
    amb_position_ambiguous = adjust_idx_rand(row['verb-amb '])[1]
    amb_position_baseline = adjust_idx_rand(row['verb-unamb '])[1]
    noun_ambiguous = adjust_idx_rand(row['np-amb '])[1]
    noun_baseline = adjust_idx_rand(row['np-umamb '])[1]

    stimuli[len(stimuli)] = {
        'source': f'classic-{condition}',
        'stimulus': " ".join(sentence),
        'baseline': " ".join(baseline_sentence),
        'disamb_position_ambiguous': disamb_position_ambiguous,
        'disamb_position_baseline': disam_position_baseline,
        'np_ambiguous': noun_ambiguous,
        'np_baseline': noun_baseline,
        'amb_position_ambiguous': amb_position_ambiguous,
        'amb_position_baseline': amb_position_baseline,
        'orig_idx': int(row.iloc[0])
    }

    if condition in ('mvrr', 'nps'):
        new_verb = MVRR_VERB if condition == 'mvrr' else NPS_VERB
        # create a new pair using the "neutral" verb for the causal analysis
        sentence[amb_position_ambiguous] = new_verb
        baseline_sentence[amb_position_baseline] = new_verb

        stimuli[len(stimuli)] = {
            'source': f'classic-{condition}_for-causal',
            'stimulus': " ".join(sentence),
            'baseline': " ".join(baseline_sentence),
            'disamb_position_ambiguous': disamb_position_ambiguous,
            'disamb_position_baseline': disam_position_baseline,
            'np_ambiguous': noun_ambiguous,
            'np_baseline': noun_baseline,
            'amb_position_ambiguous': amb_position_ambiguous,
            'amb_position_baseline': amb_position_baseline,
            'orig_idx': int(row.iloc[0])
        }

df = pd.DataFrame(stimuli).transpose().to_csv(Path(args.preproc_path))
