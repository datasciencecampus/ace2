# Matching SIC/SOC codes through ML methods


## Intent

The vague plan is to develop the ML code as a python project (sans pyspark), and if
a model is developed that functions sufficiently well, it can be imported as a submodule
into the main coding tool on ea_subsample/ea_subsample.

This way the model can be developed completely independently of and simultaneously with
the main coding tool.


## Environment Variables

| Name                  | Value                                                |
|-----------------------|------------------------------------------------------|
|PIP_INDEX_URL          |http://art-p-01/artifactory/api/pypi/yr-python/simple |	
|PIP_TRUSTED_HOST	      |art-p-01	                                             |
|PYSPARK_PYTHON	        |/opt/ons/virtualenv/miscMods_v3.05/bin/python3.6	     |
|PYSPARK_PYTHON_DRIVER  |	/opt/ons/virtualenv/miscMods_v3.05/bin/python3.6	   |
|NLTK_DATA	            |~/nltk_data                                           |
|PYTHONPATH             |/home/cdsw/ml_coding_tool                             |


## Usage

### For SOC classification
From `ml_coding_tool` run:

`python3 scripts/data_reader.py`
(For testing code, add `-lm 2000` to increase speed)

`python3 scripts/feature_engine.py`

`python3 scripts/train_test_text.py`

### For SIC classification

`python3 scripts/data_reader.py -c sic -lh industry_code -th business`
(For testing code, add `-lm 2000` to increase speed)

`python3 scripts/feature_engine.py -c sic`

`python3 scripts/train_test_text.py -d sic`

### For hyperparameter search
The experiments are controlled from config json files within `ml_coding_tool/config`.

From `ml_coding_tool` run:

`python3 scripts/hyperparameter_search.py -d <directory of cached features> -c <name of config file for experiment>`

For example for exploring Logistic Regression hyperparameters on SOC:

`python3 scripts/hyperparameter_search.py -d soc -c hyperparameter_LR.json`

### ace app
A text classification app

#### installation
To install ace:

1. pip install -e .
2. Download [Fasttext-300d](https://fasttext.cc/docs/en/english-vectors.html) and save it in models/fasttext directory  
3. Download [Glove.6B.300d](https://nlp.stanford.edu/projects/glove/) and save it in models/glove directory  
