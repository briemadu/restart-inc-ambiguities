# Interpreting Restart Incrementality - Meaning

This is the code for the analysis of the incremental construction of meaning
representations.


# Setup and Python Environment

To recreate the conda environment, run:

```bash
sh createenv.sh
``` 

Create the necessary directories:

```bash
mkdir data
mkdir outputs
mkdir outputs/embeddings
mkdir outputs/triangles
mkdir figures
```

# Data

We need two datasets:

- [Noun Compound Senses Dataset](https://github.com/marcospln/noun_compound_senses) 
by Garcia et al. (2021).
- [Classic Garden Path stimuli](https://github.com/caplabnyu/sapbenchmark)
by Huang at al. (2023)

```bash
git clone https://github.com/marcospln/noun_compound_senses
git clone https://github.com/caplabnyu/sapbenchmark
wget https://github.com/caplabnyu/sapbenchmark/raw/main/Items%20for%20all%20subsets.xlsx
```

The position of the main verb and direct object has been annotated by us for 
the SAP stimuli. We can share it upon request. The preprocessing script
relies on this extended version with more columns.

# Reproducing results

Run the scripts that preprocess data, extract the embeddings and compute the
distance triangles:

```bash
sh runscripts.sh
``` 

The plots and numbers reported in the paper are created in the three analysis*
notebooks. You must run then with the corresponding model names.

# Documentation of the Preprocessed Data

Columns and their descriptions:

- ```source```: from which dataset is the instance. Values: nnc (the noun noun compounds), classic-nps and classic-mvrr for the classic garden path data. The suffix _for-causal refers to the manipulated classic instances, wherre we replace the main verb with a reference verb removing the ambiguity but keeping the sentence structure, for comparison.
- ```stimulus```: the sentence with a temporary ambiguity.
- ```baseline```: the same sentence either with the temporary ambiguity resolved in advance (for the classic cases) or with a reference continuation with a comma (for nnc).
- ```disamb_position_ambiguous```: index (starting from 0) of the disambiguating token in the stimulus.
- ```disamb_position_baseline```: index (starting from 0) of the same disambiguating token of the stimulus but in the baseline.
- ```amb_position_ambiguous```: index (starting from 0) of the ambiguous token in the stimulus. In the classic instances, this is always the main verb (although in NP/S the direct object is also involved, but its position can be easily inferred as the next ones).
- ```amb_position_baseline```: index (starting from 0) of the same ambiguous token of the stimulus but in the baseline.
- ```orig_idx```: index in the original dataset where other metadata can be found.
- ```np_ambiguous```: position of the second noun in the stimulus.
- ```np_baseline```: position of the second noun in the baseline.

# Notes
- Parts of the code rely on models having only 12 + 1 layers.
