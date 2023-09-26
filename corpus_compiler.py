import gzip
import ujson
import orjson
import time
from bs4 import BeautifulSoup

METRIC_PRINT_AFTER_DOC_CNT = 100
MAX_DOCS = 10_000

def compile():

    record_cnt = 0
    init_start_t = time.time()
    start_t = init_start_t

    PATHNAME = "data/trec_corpus_5000.jsonl.gz"
    OUTNAME = "data/trec_corpus_5000_compiled_reduced.json"


    print(f"Starting compiler on: \"{PATHNAME}\" ...")
    doc_id_contents = {}

    with gzip.open(PATHNAME, mode="rt") as corpus:
        for document in corpus:
            doc_json = orjson.loads(document)
            doc_text = BeautifulSoup(doc_json["contents"], "html.parser").text
            doc_id_contents[doc_json["id"]] = {"title": doc_json["title"], "contents": doc_text}
            
            record_cnt += 1

            # Metrics block for analysis
            if record_cnt % METRIC_PRINT_AFTER_DOC_CNT == 0:
                end_t = time.time()
                bulk_t = end_t - start_t
                total_t = end_t - init_start_t
                print(f"- {record_cnt} records in (bulk = {bulk_t:.3f}s, total = {total_t:.3f}s)")
                start_t = end_t
            if record_cnt >= MAX_DOCS:
                break

    print(f"Compiling final results into: \"{OUTNAME}\" ...")

    with open(OUTNAME, "w") as f_doc_freq:
        ujson.dump(doc_id_contents, f_doc_freq)

    print(f"Successfully compiled into \"{OUTNAME}\": \"{PATHNAME}\"")

if __name__ == "__main__":
    compile()