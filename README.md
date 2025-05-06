# MetaSRA: normalized human sample-specific metadata for the Sequence Read Archive

This repository contains the code implementing the pipeline used to construct the MetaSRA database described in our publication: https://doi.org/10.1093/bioinformatics/btx334.

This pipeline re-annotates key-value descriptions of biological samples using biomedical ontologies.

The MetaSRA can be searched and downloaded from: http://metasra.biostat.wisc.edu/

## Dependencies

This project currently uses Python 2.7 and requires the following Python libraries:
- numpy (http://www.numpy.org)
- scipy (https://www.scipy.org/scipylib/)
- scikit-learn (http://scikit-learn.org/stable/)
- setuptools (https://pypi.python.org/pypi/setuptools)
- marisa-trie (https://pypi.python.org/pypi/marisa-trie)
- nltk (http://www.nltk.org/)
- singledispatch (https://pypi.python.org/pypi/singledispatch)
- pybktree (https://pypi.python.org/pypi/pybktree)

### Installation of dependencies

A conda environment specification is provided in the `environment.yml` file. To create the environment, run the following from the root directory of the repository:

```bash
mamba env create -n metasra-py2 -f environment.yml
```
This will create a conda environment named `metasra-py2` with the required dependencies.  To activate the environment, run:

```bash
mamba activate metasra-py2
```

The nltk library requires the punkt tokenizer to be downloaded.  To do this, run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

Finally, the PYTHONPATH environment variable must be set to include the current directory.  This can be done by running:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Setup

In order to run the pipeline, a few external resources must be downloaded and configured.  First, set up the PYTHONPATH environment variable as described above.  Then, to set up the pipeline, run the following commands:
  
    cd ./setup_map_sra_to_ontology
    ./setup.sh

This script will download the latest ontology OBO files, the SPECIALIST Lexicon files, and configure the ontologies to work with the pipeline.

## Usage

The pipeline can be run on a set of sample-specific key-value pairs
using the run_pipeline.py script. This script is used as follows:

    python run_pipeline.py <input key-value pairs JSON file>

The script accepts as input a JSON file storing a list of sets of key-value pairs.
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
