import gzip
import ujson
import orjson
import time
import lxml
import cchardet
from bs4 import BeautifulSoup
from utils import extract_keyword_set, vectorize_doc_as_inds, setup_utils, TEST_RUN

METRIC_PRINT_AFTER_DOC_CNT = 100

def preproc():

    print(f"=== PRE-PROCESSING (Test Run: {TEST_RUN}) ===")

    print("Getting relevant keywords table ...")
    setup_utils(True, False)
    dest_indices = extract_keyword_set()

    print(f"Successfully got relevant keywords table of size {len(dest_indices)} ...")

    record_cnt = 0
    init_start_t = time.time()
    start_t = init_start_t

    doc_vects = {}

    print("Opening corpus for reading ...")

    f_corpus_path = ("../data/trec_corpus_5000_compiled_reduced.json", "../data/smaller_test.json")[TEST_RUN]
    with open(f_corpus_path, "r") as f_corpus:
        corpus_dct = orjson.loads(f_corpus.read())

        for doc_id_strg in corpus_dct:
            doc_info_dct = corpus_dct[doc_id_strg]
            doc_title = doc_info_dct["title"]
            doc_text = doc_info_dct["contents"]
            doc_vects[doc_id_strg] = vectorize_doc_as_inds(doc_text, dest_indices)
    
            record_cnt += 1

            # Metrics block for analysis
            if record_cnt % METRIC_PRINT_AFTER_DOC_CNT == 0:
                end_t = time.time()
                bulk_t = end_t - start_t
                total_t = end_t - init_start_t
                print(f"- {record_cnt} records in (bulk = {bulk_t:.3f}s, total = {total_t:.3f}s)")
                start_t = end_t

    print("Saving document vectors dump ...")

    f_doc_freq_path = ("doc_vects.json.gz", "doc_vects_test.json.gz")[TEST_RUN]
    with gzip.open(f_doc_freq_path, "wt") as f_doc_vects_out:
        for k in doc_vects:
            f_doc_vects_out.write(k + "\t")
            ujson.dump(doc_vects[k], f_doc_vects_out)
            f_doc_vects_out.write("\n")
    
    print(f"Successfully finished with {record_cnt} records")