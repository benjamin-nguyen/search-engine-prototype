import orjson
import gzip
import ujson
import os
import numpy as np
from math import ceil
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from models.utils import *

CATEGORY_CNT = 46

class KMeansRanker:
    def __init__(self, search_res, use_stop_words, use_stemming, cluster_cnt = CATEGORY_CNT):

        setup_utils(use_stop_words, use_stemming)

        self.cluster_cnt = cluster_cnt
        self.search_res = search_res
        self.search_docid_set = self.__get_docid_set()

        self.dest_indices = extract_keyword_set()
        self.vect_space = MAX_VECTOR_LENGTH

        if not os.path.isfile("data/max_doc_vect.json"):
            self.__generate_max_comp_doc_vect()
    
    def rank(self, query):
        if len(self.search_docid_set) <= 1:
            # No ranking needed for {0, 1} search results
            return {}

        print(f"Now Ranking Query: '{' '.join(query)}'")

        # Loading in all relevant document vectors
        docmt_vects, docmt_vects_ids = [], []
        with gzip.open("models/doc_vects.json.gz", "rb") as fp:
            for k in fp:
                doc_id, _, vect = k.partition(b'\t')
                doc_id_ascii = doc_id.decode('ascii')
                if doc_id_ascii in self.search_docid_set:
                    # Each document ID is assigned its document vector, as an np.array
                    docmt_vects.append(vals_deserial_to_vect(orjson.loads(vect)))
                    docmt_vects_ids.append(int(doc_id_ascii))

        # Prepare K-Means, where `n_clusters` uses an arbitrary formula to estimate number of clusters given number of documents
        kmeans = KMeans(
            n_clusters = min(self.cluster_cnt, max(1, ceil(len(docmt_vects) / self.cluster_cnt))),
            random_state = 0,
            n_init = "auto"
        ).fit(docmt_vects)

        # Prediction logic
        query_vect = vals_deserial_to_vect(vectorize_doc_as_inds(query, self.dest_indices, doc_as_list=True))
        [query_cluster] = kmeans.predict([query_vect])

        # Aggregate relevant document results
        relevance_dct = {}
        for idx, doc_cluster in enumerate(kmeans.labels_):
            if doc_cluster == query_cluster:
                # Query and document clusters match (come from the same cluster), use euclidean distance for further scoring
                relevance_dct[docmt_vects_ids[idx]] = euclidean(query_vect, docmt_vects[idx])
            else:
                # In classification, we want positive scores for matching clusters, hence we return score of 0 for non-matching clusters
                relevance_dct[docmt_vects_ids[idx]] = 0

        # Use the pre-computed euclidean distance to correlate nearest distance with relevance in a scoring S : (0, 1]
        highest_dist = max(relevance_dct.values())
        for doc_id in relevance_dct:
            if relevance_dct[doc_id] > 0:
                relevance_dct[doc_id] = 1 - relevance_dct[doc_id] / highest_dist

        return relevance_dct

    def __get_doc_relevance(self, query_cluster, doc_cluster, query_vect, doc_vect):
        if doc_cluster == query_cluster:
            return euclidean(query_vect, doc_vect)
        return 0

    def __get_docid_set(self):
        docid_set = set()
        for docid, _ in self.search_res[1]:
            docid_set.add(str(docid))
        return docid_set

    def __generate_max_comp_doc_vect(self):
        print("Generating max-component document vector ...")
        max_vect = {}
        with gzip.open("models/doc_vects.json.gz", "rb") as fp:
            for k in fp:
                _, _, vect = k.partition(b'\t')
                for a, b in orjson.loads(vect).items():
                    max_vect[a] = max(max_vect.get(a, 0), b)

        print("Outputting max-component document vector ...")
        with open("data/max_doc_vect.json", "w") as fout:
            ujson.dump(max_vect, fout)
        print("Output succeeded")