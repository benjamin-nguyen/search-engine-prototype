﻿# Search Engine Prototype 

## Description

This search engine system is built with inverted indexing, a GUI/TUI, and various ranking algorithms (TF, BM25, K-means, KNN).

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
   
## Abstract

Determining the optimal ranking algorithm for a search engine involves evaluating various models for efficiency and accuracy. This study examines several algorithms, including Term Frequency (TF), BM25, K-Nearest Neighbors (KNN), and K-Means, implemented in Python. By comparing their performance in terms of runtime and search result relevance, we identified that the BM25 model outperformed the others, providing the most accurate results efficiently.

## Introduction

Building a search engine from scratch is complex due to the numerous components and factors involved. A critical aspect is processing search queries and returning the most relevant documents. This paper explores various algorithms for ranking and retrieving documents, including TF, BM25, KNN, and K-Means. By implementing and comparing these models, we aim to identify the most effective approach for both efficiency and accuracy in search results.

## Architecture

The search engine was implemented using a conventional inverted index, relevance ranking models, and a web application interface. Several optimizations were applied, such as storing document vectors and precompiled raw documents as intermediate files. These optimizations reduced the time required to build the inverted index, improved document access time for BM25, and eliminated the need for on-the-fly document vector calculations for K-Means and KNN.

![333892892-200f0f71-27b7-44d9-be78-c8479b040872 (1)](https://github.com/benjamin-nguyen/search-engine-prototype/assets/55249079/a483c97d-c5aa-4386-808e-1e77dec47611)


## Discussion

Four different models were used: TF, BM25, KNN, and K-Means, detailed as follows:

**Term Frequency (TF):**
This model counts the occurrences of a term in a document and ranks documents based on term frequency. The document with the highest frequency is displayed first.

```python
if mode == 0:
    acc_search_res[1].sort(key=lambda doc: doc[1]['f'], reverse=True)
```

**BM25:**
BM25 applies the TF model and Inverse Document Frequency (IDF) to rank documents. The ranking formula involves several parameters, including term frequency, document length, and average document length. The detailed BM25 formulas are provided, outlining the steps to calculate document ranks.

1. $$\sum \log \left( \frac{(r + 0.5)/(R - r + 0.5)}{(n - r + 0.5)/(N - n - R + r + 0.5)} \right)$$

2. $$\frac{(k1 + 1)f}{K + f}$$

3. $$\frac{(K2 + 1)Q}{k2 + Q}$$

4. $$K = k1 \left( (1 - b) + b \cdot \frac{dl}{avdl} \right)$$

**KNN (K-Nearest Neighbors):**
KNN is a supervised machine learning algorithm that assumes similar objects are in close proximity. The algorithm ranks documents based on the distances between the query vector and document vectors, identifying the "K" closest documents for classification.

![KNN](https://github.com/benjamin-nguyen/search-engine-prototype/assets/55249079/958c879f-dad9-495d-a2d5-7ff37bb90b6e)

**K-Means:**
K-Means is an unsupervised machine learning algorithm that clusters data into samples containing centroids. The algorithm iteratively assigns samples to the nearest centroids and recalculates centroids until minimal movement occurs.

![K-Means](https://github.com/benjamin-nguyen/search-engine-prototype/assets/55249079/4d4be34f-43f1-4fb6-bba0-62ec366c8feb)


## Demonstration
Users can customize options for stop words, stemming, document display limits, and search queries. The results vary across algorithms due to their unique scoring methods. The runtime of each algorithm is compared, demonstrating BM25's superior performance with an average processing time of 2-4 seconds, faster than other models except for TF.

![Animation](https://github.com/benjamin-nguyen/search-engine-prototype/assets/55249079/75dfa1c0-a6d0-4855-a48e-6256c91a6c0b)

### Example Queries (stop words, no stemming):

**TF**:

| Query       | df   | Time Taken |
|-------------|------|------------|
| “Soup”      | 67   | 2.030s     |
| “Software”  | 555  | 2.110s     |
| “England”   | 1548 | 0.922s     |

- Uses only search results to rank
- Affected solely by I/O time from retrieving search results
- Despite simplicity and speed, skews relevance by occurrences in the document

**BM25**:

| Query       | df   | Time Taken |
|-------------|------|------------|
| “Soup”      | 67   | 1.974s     |
| “Software”  | 555  | 2.275s     |
| “England”   | 1548 | 3.765s     |

- Higher df -> more calculations -> slower runtime
- Performance dictated by I/O from loading document lengths
- Decent solution where df is kept low

**K-Means**:

| Query       | df   | Time Taken |
|-------------|------|------------|
| “Soup”      | 67   | 3.818s     |
| “Software”  | 555  | 6.367s     |
| “England”   | 1548 | 10.342s    |

- Higher df * high number of clusters (k=46) = slow runtime
- Baseline ~3.0s per search due to loading document vectors from I/O
- More suitable for datasets using smaller k-values

**KNN**:

| Query       | df   | Time Taken |
|-------------|------|------------|
| “Soup”      | 67   | 3.841s     |
| “Software”  | 555  | 3.932s     |
| “England”   | 1548 | 3.735s     |

- Performance is relatively constant
- Baseline I/O time for loading document vectors
- Constant number of vector distance calculations with query
- Constant `K = 4,965` (avg. docs per `reldocs.tsv` category)
- Suitable for scalability of accepting any df

![image](https://github.com/benjamin-nguyen/search-engine-prototype/assets/55249079/b9f31734-9ae3-4241-97b5-292607a84cff)

## Limitations

The search engine's performance may decline with limited computational power or outdated hardware. Additionally, the relatively small database used in this study contrasts with the extensive databases of commercial search engines like Google.

## Conclusions

The BM25 model is the most effective ranking algorithm for a search engine, offering accurate results without the unpredictability of machine learning models like K-Means and KNN. BM25's reliance on human intervention ensures consistent and reliable outcomes.

