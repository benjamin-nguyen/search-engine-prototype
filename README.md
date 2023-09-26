# Search Engine Prototype

## Description

Search engine system built with inverted indexing, a web application interface, and algorithms for ranking (TF, BM25, K-means, KNN)

## Setup

The following steps use **Python 3.10.2**, not fully tested:

1. Ensure that all Python packages from each `*.py` file are installed beforehand
   - Cython requires **Microsoft Visual C++ 14.0** prior to installation
   - Includes: `cython Flask beautifulsoup4 ujson orjson lxml cchardet scikit-learn scipy numpy nltk`
2. Enter the project root directory
3. Have `trec_corpus_5000.jsonl.gz` (A1 data) downloaded into the `data` folder
   - Download this separately, from [here](https://drive.google.com/file/d/1FIrsU9X2JmgnT4imsZkYHFv_zEVHUDoL/view?usp=sharing)
4. Run `corpus_compiler.py` to get a sanitized version of the corpus
5. Run `python setup.py build_ext --inplace` for inverted index Cython files
6. Run `invert_run.py` to run entire inverted index
7. Enter the models directory
8. Run `python kmeans_setup.py build_ext --inplace` for K-means Cython files
9. Run `kmeans_prep_run.py` to generate a document vectors file
10. Move back into project root directory
11. You can do one of the following to run searches:
    - Type in `python -m flask run` to start querying through a GUI
    - Type in `python search.py` to start querying through a TUI
