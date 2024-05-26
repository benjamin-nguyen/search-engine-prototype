import os
import orjson
import random
import numpy as np
from math import log2

TEST_RUN = False
MAX_VECTOR_LENGTH = 100_000
TREC_CORPUS_5000_DOC_CNT = 212_651

master_doc_freq = None
idf_cache = None
sanitize_table = {ord(k): None for k in '0123456789[].,";/{}!()*_:+<>?=@&-|†↑'}

def setup_utils(use_stop_words, use_stemming):
    global master_doc_freq, idf_cache

    if TEST_RUN:
        print("Loading Test Document Frequencies Table ...")
        doc_freq_test_f_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../..", "data", "doc_freq_test.json"))
        master_doc_freq = orjson.loads(open(doc_freq_test_f_path).read())
    else:
        out_folders = {
            (True, True): "out_stop_stem",
            (False, True): "out_nostop_stem",
            (True, False): "out_stop_nostem",
            (False, False): "out_nostop_nostem",
        }
        out_folder = out_folders[(use_stop_words, use_stemming)]
        print("Loading Document Frequencies Table ...")
        doc_freq_f_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../..", out_folder, "doc_freq.json"))
        master_doc_freq = orjson.loads(open(doc_freq_f_path).read())

    print("Loading IDF Cache ...")
    idf_cache = {v: log2(TREC_CORPUS_5000_DOC_CNT / v) for v in master_doc_freq.values()}

def extract_keyword_set(max_keyword_cnt = MAX_VECTOR_LENGTH):
    keywords = set()
    for word, _ in sorted(master_doc_freq.items(), key = lambda v: -v[1]):
        if len(keywords) >= max_keyword_cnt:
            break
        if word != '':
            keywords.add(word)

    if len(keywords) < max_keyword_cnt and not TEST_RUN:
        f_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../..", "data", "train_topics_keywords.tsv"))

        with open(f_path, encoding="utf8") as f_keywords:
            for line in f_keywords.read().strip().split('\n'):
                _, _, new_keywords = line.split('\t')
                for word in new_keywords.split(','):
                    if len(keywords) >= max_keyword_cnt:
                        break
                    word_low = word.lower()
                    if word_low != '':
                        keywords.add(word_low)

    dest_indices = {term: idx for idx, term in enumerate(sorted(keywords))}
    return dest_indices

def vectorize_doc_as_inds(doc_text, dest_indices, doc_as_list = False):
    doc_vect, terms_lst = {}, doc_text if doc_as_list else doc_text.split()
    if len(terms_lst) == 0:
        return {}

    for term in terms_lst:
        term_lower = term.lower()
        term_sanitized = term_lower.translate(sanitize_table)
        if term_sanitized in dest_indices:
            # Below, key in the dictionary is _meant_ to be a numerical string (leave it as it is)
            vect_idx = str(dest_indices[term_sanitized]) 
            if vect_idx in doc_vect:
                doc_vect[vect_idx][1] += 1
            else:
                doc_vect[vect_idx] = [term_sanitized, 1]

    # Transform term count vector into TF-IDF vector
    doc_tfidf_vect = {}
    for vect_idx, (term, term_cnt) in doc_vect.items():
        doc_tfidf_vect[vect_idx] = term_cnt / len(terms_lst) * idf_cache[master_doc_freq.get(term, 0)]
    return doc_tfidf_vect

def vectorize_doc_as_nparr(doc_text, dest_indices):
    doc_vect, terms_lst = {}, doc_text.split()
    if len(terms_lst) == 0:
        return {}

    for term in terms_lst:
        term_lower = term.lower()
        term_sanitized = term_lower.translate(sanitize_table)
        if term_sanitized in dest_indices:
            vect_idx = dest_indices[term_sanitized]
            if vect_idx in doc_vect:
                doc_vect[vect_idx][1] += 1
            else:
                doc_vect[vect_idx] = [term_sanitized, 1]

    # Transform term count vector into TF-IDF vector
    doc_tfidf_vect = np.zeros(MAX_VECTOR_LENGTH)
    for vect_idx, (term, term_cnt) in doc_vect.items():
        doc_tfidf_vect[vect_idx] = term_cnt / len(terms_lst) * idf_cache[master_doc_freq.get(term, 0)]
    return doc_tfidf_vect

def vals_deserial_to_vect(vals_dict):
    vect = np.zeros(MAX_VECTOR_LENGTH)
    for word_idx, vect_val in vals_dict.items():
        idx = int(word_idx)
        if idx >= len(vect):
            continue
        vect[int(idx)] = vect_val
    return vect

def doc_vect_as_true_vect(doc_vect):
    vect = [0] * MAX_VECTOR_LENGTH
    for k in doc_vect:
        idx = int(k)
        if idx >= len(vect):
            continue
        vect[idx] = doc_vect[k]
    return vect

def get_randomized_vector(max_doc_vect):
    vect = np.zeros(MAX_VECTOR_LENGTH)
    for i in range(len(vect)):
        vect[i] = random.random() * max_doc_vect[i]
    return vect

def cosine_sim_from_sparse_vect_dicts(u, v):
    """
    Each document is a long, sparse vector, so here we represent
    these long sparse vectors as dictionaries where the key matches
    the array index for all non-zero elements. Here, we simply take
    the cosine similarity as if `u` and `v` were two vectors represented
    as these dictionaries.

    This is also convenient for the way we store document vectors in
    order to save space.
    """
    uv = 0
    uu = 0
    vv = 0
    for k in u:
        uu += u[k] * u[k]
    for k in v:
        vv += v[k] * v[k]
        uv += u.get(k, 0) * v[k]
    uv /= np.sqrt(uu)
    uv /= np.sqrt(vv)
    return uv

def cosine_sim_from_ndarrs(u, v):
    """
    Typical cosine similarity for np.ndarray types, incase above
    function is not good enough / requirements change?
    """
    uv = 0
    uu = 0
    vv = 0
    for k in range(len(u)):
        uu += u[k] * u[k]
    for k in range(len(v)):
        vv += v[k] * v[k]
        uv += u[k] * v[k]
    uv /= np.sqrt(uu)
    uv /= np.sqrt(vv)
    return uv