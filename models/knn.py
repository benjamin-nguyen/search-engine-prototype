import orjson
import gzip
import random
import time
import ujson
import os
import numpy as np
from math import pow
from scipy.spatial.distance import euclidean
from models.utils import *

BEST_K = 4_965 # This is the average number of documents for each of the `reldocs` categories

class KNNRanker:
    def __init__(self, search_res, use_stop_words, use_stemming, k = BEST_K):
        self.k = k
        self.search_res = search_res
        self.topic_map = {}

        setup_utils(use_stop_words, use_stemming)

        self.dest_indices = extract_keyword_set()
        self.search_docid_set = self.__get_docid_set()
        self.docid_to_topic_map = self.__get_classification_data()

    def rank(self, query):

        # Loading in all relevant document vectors
        docmt_vects = {}
        with gzip.open("models/doc_vects.json.gz", "rb") as fp:
            for k in fp:
                doc_id, _, vect = k.partition(b'\t')
                doc_id_ascii = doc_id.decode('ascii')
                if doc_id_ascii in self.search_docid_set:
                    # Each document ID is assigned its cluster number (initially -1) and JSON vector magnitudes 
                    docmt_vects[doc_id_ascii] = vals_deserial_to_vect(orjson.loads(vect))

        # Load in query vector
        query_vect = vals_deserial_to_vect(vectorize_doc_as_inds(query, self.dest_indices, doc_as_list=True))
        
        # Get all distances from query to documents
        query_doc_distances = []
        for doc_id in docmt_vects:
            # Use scipy's `euclidean` here to make this basic distance calculation faster 
            query_doc_distances.append((euclidean(query_vect, docmt_vects[doc_id]), doc_id))
        query_doc_distances.sort()

        # Get the `k` "closest distance" documents and their categories
        k_nearest_labels, k_nearest = {}, query_doc_distances[:self.k]
        for _, doc_id in k_nearest:
            relevant_topic_id = self.docid_to_topic_map[doc_id]
            k_nearest_labels[relevant_topic_id] = k_nearest_labels.get(relevant_topic_id, 0) + 1

        # Get mode of `k_nearest_labels` to classify query
        query_class_id = sorted(k_nearest_labels.items(), key = lambda v: v[1])[-1][0]

        # Extract all relevant docs by their ID
        relevant_docs_by_id = set()
        for doc_id in self.docid_to_topic_map:
            if self.docid_to_topic_map[doc_id] == query_class_id:
                relevant_docs_by_id.add(doc_id)
        
        # Relevance would return 1 if the document part of the query's class, otherwise, 0
        relevance_dct = {int(doc_id): int(doc_id in relevant_docs_by_id) for doc_id in docmt_vects}

        # Relevance is assigned inverse query distance from doc. for all scores of 1, so nearest will come first
        query_doc_distances_dct = {int(doc_id): dist for dist, doc_id in query_doc_distances}
        for k in relevance_dct:
            relevance_dct[k] /= query_doc_distances_dct[k]
        
        print(max(relevance_dct.values()), min(relevance_dct.values()))

        return relevance_dct

    def __get_classification_data(self):
        docid_to_topic_map = {}
        with open("data/train_topics_reldocs.tsv", "r") as fc:
            for k in fc:
                topic_id, topic, reldocs = k.strip().split('\t')
                self.topic_map[topic_id] = topic
                reldocs = reldocs.split(',')
                for reldoc_id in reldocs:
                    docid_to_topic_map[reldoc_id] = topic_id
        return docid_to_topic_map

    def __get_docid_set(self):
        docid_set = set()
        for docid, _ in self.search_res[1]:
            docid_set.add(str(docid))
        return docid_set