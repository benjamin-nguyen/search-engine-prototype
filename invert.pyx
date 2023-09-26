import gzip
import ujson
import orjson
import time
import re
import os.path
from collections import defaultdict, deque
from nltk.corpus import stopwords
from nltk.stem import porter
from bs4 import BeautifulSoup
from string import ascii_lowercase

METRIC_PRINT_AFTER_DOC_CNT = 100
UPDATE_EVERY_N_DOCUMENTS = 50_000
INC_POST_INDEX_EVERY_N_DOCUMENTS = 50_000
CONTEXT_WINDOW_LEN = 10
CONTEXT_WINDOW_LEN_HALF = 5

porter_stemmer = porter.PorterStemmer()
stop_words_en = set(stopwords.words('english'))

sanitize_table = {ord(k): None for k in '0123456789[].,";/{}!()*_:+<>?=@&-|†↑'} 

output_path = None

res_doc_freq = defaultdict(int)
res_post_buckets = defaultdict(dict)

cdef int posting_idx = 1



def safe_nested_mkdir(*dirs):
    """
    Safely makes a nested directory.

    Example:

    >>> safe_nested_mkdir("dir1", "dir2", "dir3")

    1. Makes folder at "dir1"
    2. Makes folder at "dir1/dir2"
    3. Makes folder at "dir1/dir2/dir3"
    """
    if not os.path.isdir("/".join(dirs)):
        for i in range(1, len(dirs) + 1):
            try:
                os.mkdir("/".join(dirs[:i]))
            except OSError:
                pass
    
    
    
def merge_posting_json(postings_a, postings_b):
    """
    This function merges the JSON results of two posting lists.

    Example:

    pst_a = {"apple": {"1": {"t": "Article1", "f": 2, "p": [5, 12]}}, "banana": {"1": {"t": "Article4", "f": 1, "p": [8]}}}
    pst_b = {"apple": {"2": {"t": "Article2", "f": 3, "p": [7, 19, 43]}, "3": {"t": "Article3", "f": 1, "p": [2]}}, "canada": {"4": {"t": "Article5", "f": 1, "p": [81]}}}
    pst_ab = merge_posting_json(pst_a, pst_b)
    all((len(pst_ab) == 3, len(pst_ab["apple"]) == 3, len(pst_ab["banana"]) == 1))
    """
    for k in postings_b:
        postings_a[k] = postings_a.get(k, {}) | postings_b[k]
    return postings_a



def send_res_to_files():
    """
    This function:
        1. Takes `res_doc_freq` and creates the "dictionary" out of it
        2. Takes `res_post_buckets` and creates a bunch of buckets (by first
           letter of term) and creates the "intermediate posting lists" out of it
    """
    global res_doc_freq
    global res_post_buckets

    # Outputting of posting list buckets (a -> z)
    for letter in ascii_lowercase:
        safe_nested_mkdir(output_path, letter)
        postings_fname = f"{output_path}/{letter}/{posting_idx}.json"
        if os.path.isfile(postings_fname):
            with open(postings_fname, "r") as f_curr_postings:
                postings = orjson.loads(f_curr_postings.read())
        else:
            postings = {}
        curr_postings_len = len(postings)

        postings = merge_posting_json(postings, res_post_buckets.get(letter, {}))

        if curr_postings_len != len(postings):
            print(f"* Key count for '{letter}' : {curr_postings_len} -> {len(postings)}")
        with open(postings_fname, "w") as f_new_postings:
            ujson.dump(postings, f_new_postings)
    res_post_buckets = defaultdict(dict)

    # Outputting of document frequency file
    safe_nested_mkdir(output_path)
    doc_freq_fname = f"{output_path}/doc_freq.json"
    if os.path.isfile(doc_freq_fname):
        with open(doc_freq_fname, "r") as f_curr_doc_freq:
            doc_freq = orjson.loads(f_curr_doc_freq.read())
    else:
        doc_freq = {}
    curr_doc_freq_len = len(doc_freq)

    for key in res_doc_freq:
        doc_freq[key] = doc_freq.get(key, 0) + res_doc_freq[key]

    if curr_doc_freq_len == len(doc_freq):
        print(f"* Key count for doc_freq : No changes")
    else:
        print(f"* Key count for doc_freq : {curr_doc_freq_len} -> {len(doc_freq)}")
    with open(doc_freq_fname, "w") as f_doc_freq:
        ujson.dump(doc_freq, f_doc_freq)
    res_doc_freq = defaultdict(int)



def build_invert(param_output_path, use_stop_words = True, use_stemming = True):
    """
    This function is the main logic to building the inverted index. The corpus
    is read, parsing information into and accumulating `res_doc_freq` and
    `res_post_buckets`. Periodically, these two dictionaries have all of their
    contents emptied into (or merged with existing) output files.
    """
    global res_doc_freq
    global res_post_buckets
    global posting_idx
    global output_path

    cdef int record_cnt = 0
    cdef double init_start_t = time.time()
    cdef double start_t = time.time()
    cdef double end_t, bulk_t, total_t
    not_use_stop_words = not use_stop_words
    output_path = param_output_path

    print(f"Running invert with props (stop_words={use_stop_words}, stemming={use_stemming}, outpath=\"{output_path}\")")

    with open("data/trec_corpus_5000_compiled_reduced.json", "r") as f_corpus:
        corpus_dct = orjson.loads(f_corpus.read())

        for doc_id_strg in corpus_dct:
            doc_info_dct = corpus_dct[doc_id_strg]

            doc_id = int(doc_id_strg)
            doc_title = doc_info_dct["title"]
            doc_text = doc_info_dct["contents"]
            words_fnd = set()

            # Build initial context window
            ctx_window = deque()
            ctx_window_iter = re.finditer(r'\S+', doc_text)
            while len(ctx_window) < CONTEXT_WINDOW_LEN:
                try:
                    ctx_window.append(next(ctx_window_iter).group(0))
                except StopIteration:
                    break

            word_cnt = 0
            for match_iter in re.finditer(r'\S+', doc_text):
                # Start updating context window after reaching window half-point
                try:
                    if word_cnt > CONTEXT_WINDOW_LEN_HALF:
                        nxt_ctx_window = next(ctx_window_iter).group(0)
                        ctx_window.append(nxt_ctx_window)
                        ctx_window.popleft()
                except StopIteration:
                    pass
                word_cnt += 1

                # Word tokenizing and position-finding in `doc_text`
                word_raw = match_iter.group(0)
                word_low = word_raw.lower()
                word_sanitized = word_low.translate(sanitize_table)
                if not word_sanitized:
                    continue
                word_position = match_iter.start()
                if use_stemming:
                    word_stem = porter_stemmer.stem(word_sanitized)
                    if not word_stem:
                        continue
                else:
                    word_stem = word_sanitized
                if not_use_stop_words and word_stem in stop_words_en:
                    continue
                word_first_alpha = word_sanitized[0]

                # Update term's frequency and position in document
                bucket_by_alpha = res_post_buckets[word_first_alpha]
                bucket_by_alpha[word_stem] = bucket_by_alpha.get(word_stem, {})
                term_postings = bucket_by_alpha[word_stem]
                term_postings[doc_id] = \
                    term_postings.get(doc_id, {
                        "t": doc_title,
                        "s": " ".join(ctx_window),
                        "f": 0,
                        "p": []
                    })
                term_postings[doc_id]["f"] += 1
                term_postings[doc_id]["p"].append(word_position)

                # Update document frequency
                if word_stem not in words_fnd:
                    res_doc_freq[word_stem] += 1
                    words_fnd.add(word_stem)
            
            record_cnt += 1

            # Metrics block for analysis
            if record_cnt % METRIC_PRINT_AFTER_DOC_CNT == 0:
                end_t = time.time()
                bulk_t = end_t - start_t
                total_t = end_t - init_start_t
                print(f"- {record_cnt} records in (bulk = {bulk_t:.3f}s, total = {total_t:.3f}s)")
                print(f"  - doc_freq_len = {len(res_doc_freq)}\n  - post_buckets_len = {len(res_post_buckets)}")
                start_t = end_t

            # Periodic update of output files
            if record_cnt % UPDATE_EVERY_N_DOCUMENTS == 0:
                send_res_to_files()
            if record_cnt % INC_POST_INDEX_EVERY_N_DOCUMENTS == 0:
                posting_idx += 1

    # Send all stored changes into output files
    send_res_to_files()

    print("The inverted index has successfully finished.")