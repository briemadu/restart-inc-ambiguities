#!/bin/bash

python3 preprocess_stimuli.py

python3 extract_embeddings.py -model bert
python3 extract_embeddings.py -model roberta
python3 extract_embeddings.py -model gpt2
python3 extract_embeddings.py -model opt

python3 compute_triangles.py -model bert
python3 compute_triangles.py -model roberta
