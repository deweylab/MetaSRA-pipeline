## Other notes for getting things running.

### Cloning the repository

Note that since we use a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), it needs to be explicitly updated after cloning:

```
git submodule init
git submodule update
```

Alternatively, the repository can be cloned with submodules using the `--clone-submodules` option.

### Python version

We're running this on python 2.7, with pip 19.1.1.

### Enviornment variables

We've added `init-environment.sh` which will append the pythonpath correctly if run from the repo root.
Note that we added the directory `pip-packages`, created by the next step, to include our installed packages.

`init-environment.sh` will also point `$NLTK_DATA` to a local `nltk_data` folder which we'll need below.

### Dependencies

We put the pip-installable requirements in `requirements.txt`.
To force pip to install versions compatible with python 2.7 we run it as follows:

```
python -m pip install --python-version 2.7 --no-deps --target pip-packages -r requirements.txt
```

### Setup

First, we need to download data for the NLTK punctuation tokenizer. The old version of NLTK won't download this correctly, so we use manual download, using the `download-tokenizer.sh` script.

```
source download-tokenizer.sh
```

Next, follow the instuctions in the main README

```
cd ./setup_map_sra_to_ontology
./setup.sh
```


