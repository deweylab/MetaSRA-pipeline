# MetaSRA: normalized human sample-specific metadata for the Sequence Read Archive

This repository contains the code implementing the pipeline used to construct the MetaSRA database described in our publication: https://doi.org/10.1093/bioinformatics/btx334.

This pipeline re-annotates key-value descriptions of biological samples using biomedical ontologies.

The MetaSRA can be searched and downloaded from: https://metasra.platformx.wisc.edu/

## Installation

### Dependencies

This project requires Python 3.10 or later and the following Python libraries:
- numpy (http://www.numpy.org)
- scipy (https://www.scipy.org/scipylib/)
- scikit-learn (http://scikit-learn.org/stable/)
- setuptools (https://pypi.python.org/pypi/setuptools)
- marisa-trie (https://pypi.python.org/pypi/marisa-trie)
- nltk (http://www.nltk.org/)
- pybktree (https://pypi.python.org/pypi/pybktree)

The `metasra setup` command also requires `curl` to be available on `PATH`.

### Installation with pip

From the repository root, install the package to make the `metasra` command available:

```bash
python -m pip install .
```

For development purposes, use the -e option to install the package in "editable" mode:

```bash
python -m pip install -e .
```

### NLTK setup

The NLTK library requires the `punkt_tab` tokenizer data and stopwords to be downloaded. To do this, run:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Setup

In order to run the pipeline, a few external resources must be downloaded and configured.  To set up the pipeline, run the following command:
  
    metasra setup metasra_ref

Where `metasra_ref` is the name (of your choice) of the directory into which the resources will be downloaded.  This script will download the latest ontology OBO files, the SPECIALIST Lexicon files, and configure the ontologies to work with the pipeline.

## Usage

The pipeline can be run on a set of sample-specific key-value pairs
using the `metasra run` command:

    metasra run -r metasra_ref <input key-value pairs JSON file>

The command accepts a JSON file containing a list of sample metadata objects, each represented as key-value pairs.
For example, the pipeline will accept a file with the following content:

    [
      {   
        "ID": "P352_141",
        "age": "48",
        "bmi": "24",
        "gender": "female",
        "source_name": "vastus lateralis muscle_female",
        "tissue": "vastus lateralis muscle"
      },
      {   
        "ID": "P352_141",
        "age": "29",
        "bmi": "30",
        "gender": "male",
        "source_name": "vastus lateralis muscle_female",
        "tissue": "vastus lateralis muscle"
      }
     ]

For quick testing the above content can be found in the `example.json` file.
