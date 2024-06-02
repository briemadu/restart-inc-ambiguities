# Interpreting Restart Incrementality - Dependency Parsing

This is the code for the analysis of the incremental construction of dependency parses. We assume that the conda environment has been created following `analysis_meaning/README.md`.

# Setup
``` python
mkdir figures
mkdir outputs
```

# Requirements
* Install python3 requirements: `pip install -r requirements.txt`
* We use forks of [SuPar](https://github.com/yzhangcs/parser/tree/main) and [DiaParser](https://github.com/Unipisa/diaparser/tree/master) available under `parser` and `diaparser`, respectively. Please follow the corresponding instructions to install both libraries from **the source**.

# Data
We assume that the file `analysis_meaning/outputs/preprocessed_stimuli.csv` is already available. We can then build the data needed for analysis using `build_parse_obj.py`:

``` python
python build_parse_obj.py --model <model_name>
```

# Analysis
The code needed to reproduce the analyses is available in `analysis-dependency.ipynb` and `analysis-dep-variation.ipynb`.