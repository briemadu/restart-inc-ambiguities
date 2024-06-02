#!/bin/bash

conda create -n restart-inc python=3.9.18
conda activate restart-inc
pip install transformers
pip install torch
pip install seaborn
pip install h5py
pip install openpyxl
pip install scikit-learn