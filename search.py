import time
import orjson
import os
from nltk.corpus import stopwords
from nltk.stem import porter
from models.knn import KNNRanker
from models.kmeans import KMeansRanker
from models.bm25 import BM25Ranker

porter_stemmer = porter.PorterStemmer()
stop_words_en = set(stopwords.words('english'))

def get_postings(query, query_doc_freq, out_folder):
    if query_doc_freq == 0:
        return {}
    first_letter, search_idx, acc_postings = query[0], 1, {}
    while True:
        postings_file_name = f"{out_folder}/{first_letter}/{search_idx}.json"
        if not os.path.isfile(postings_file_name):
            break
        with open(postings_file_name, "rb") as f_postings:
            postings_json = orjson.loads(f_postings.read())
            if query in postings_json:
                for k, v in postings_json[query].items():
                    acc_postings[int(k)] = v
                if len(acc_postings) >= query_doc_freq:
                    break
        search_idx += 1
    return acc_postings

def test_and_search(query, doc_freq_table, out_folder):
    query, start_t = query.lower(), time.time()
    query_doc_freq = doc_freq_table.get(query, 0)
    res = [
        query_doc_freq,
        get_postings(query, query_doc_freq, out_folder),
    ]
    bulk_t = time.time() - start_t
    return res, bulk_t

def prompt_yn(prompt, allowed = set(['y', 'Y', 'n', 'N'])):
    uinput = None
    while uinput not in allowed:
        uinput = input(prompt)
    return uinput.lower() == "y"

def pretty_print(search_res, orig_query, limit):
    """
    This function pretty-prints the query results. It merely prints out
    all of the results from `search_res` without any other operations, such
    as abiding the `limit` constraint.
    """
    doc_freq, queried_docs = search_res
    print("Query Results:")
    print(f"    Original Query: term = \"{orig_query}\" | result_cnt = {limit}")
    print("    Document Frequency:", doc_freq)
    print("    Document Results:", "N/A" if len(queried_docs) == 0 else "")

    for doc_idx, (doc_id, doc) in enumerate(queried_docs, start=1):
        print(f"    [ {doc_idx} ]")
        print("        Document ID:", doc_id)
        print("        Document Title:", doc['t'])
        print("        Context Window:", f"\"{doc['s']}\"")
        print("        Term Frequency:", doc['f'])
        print("        Term Positions:", doc['p'])
        if doc_idx >= limit:
            break

def main():

    # Initial bucket selection
    out_folders = {
        (True, True): "out_stop_stem",
        (False, True): "out_nostop_stem",
        (True, False): "out_stop_nostem",
        (False, False): "out_nostop_nostem",
    }
    use_stop_words = prompt_yn("[Search] Use Stop Words? (Y/N): ")
    use_stemming = prompt_yn("[Search] Use Stemming? (Y/N): ")
    out_folder = out_folders[(use_stop_words, use_stemming)]

    # Loading in document frequencies
    print("Loading Document Frequencies Table ...")
    try:
        with open(f"{out_folder}/doc_freq.json", "rb") as f_doc_freq:
            doc_freq_table = orjson.loads(f_doc_freq.read())
    except FileNotFoundError as error:
        print(error)
        exit()
        
    total_t = 0
    query_cnt = 0

    # Querying document limit
    try:
        doc_limit = int(input("[Search] Enter a query document print limit (defaults to 5): "))
    except ValueError:
        doc_limit = 5

    while True:
        query = input("[Search] Enter a query (press enter or write \"ZZEND\" to quit): ").strip()
        orig_query = query
        query = query.split()

        # Exit statements and option setup
        if not query:
            break 
        elif query[0].upper() == 'ZZEND':
            break
        elif use_stemming:
            for t in range(len(query)):
                query[t] = porter_stemmer.stem(query[t])

        # Querying the search mode
        try:
            mode = int(input("  [Search] Enter a search mode (TF: 0, BM25: 1, K-Means: 2, KNN: 3): "))
        except ValueError:
            mode = 0

        # Get all accumulated search results for entire query
        query_bulk_t, acc_search_res = 0, [0, {}]
        for term in query:
            search_res, term_bulk_t = test_and_search(term, doc_freq_table, out_folder)
            acc_search_res[1] |= search_res[1]
            query_bulk_t += term_bulk_t
        acc_search_res = [len(acc_search_res[1]), list(acc_search_res[1].items())]

        # Sort accumulated search results based on search mode
        start_sort_t = time.time()
        if mode == 0:
            acc_search_res[1].sort(key=lambda doc: doc[1]['f'], reverse=True)
        elif mode in {1, 2, 3}:
            if mode == 1:
                # BM25
                ranker = BM25Ranker(acc_search_res)
            elif mode == 2:
                # KMeans
                ranker = KMeansRanker(acc_search_res, use_stop_words, use_stemming)
            elif mode == 3:
                # KNN
                ranker = KNNRanker(acc_search_res, use_stop_words, use_stemming)
            doc_rel_scores = ranker.rank(query)
            acc_search_res[1].sort(key=lambda doc: doc_rel_scores.get(doc[0], 0), reverse=True)
        end_sort_t = time.time()
        query_bulk_t += end_sort_t - start_sort_t

        # Results processing
        pretty_print(acc_search_res, orig_query, doc_limit)
        print(f'\n* Query took {query_bulk_t:.3f}s')
        query_cnt += 1
        total_t += query_bulk_t

    if query_cnt > 0:
        avg_t = total_t / query_cnt
    else:
        avg_t = 0
    print(f"* {query_cnt} queries in (total = {total_t:.3f}s, avg = {avg_t:.3f}s)")

def search(doc_limit, use_stop_words, use_stemming, query, mode):

    # Initial bucket selection
    out_folders = {
        (True, True): "out_stop_stem",
        (False, True): "out_nostop_stem",
        (True, False): "out_stop_nostem",
        (False, False): "out_nostop_nostem",
    }
    use_stop_words = use_stop_words.lower() == 'y'
    use_stemming = use_stemming.lower() == 'y'
    out_folder = out_folders[(use_stop_words, use_stemming)]

    # Loading in document frequencies
    try:
        with open(f"{out_folder}/doc_freq.json", "rb") as f_doc_freq:
            doc_freq_table = orjson.loads(f_doc_freq.read())
    except FileNotFoundError as error:
        print(error)
        exit()
        
    total_t = 0

    # Querying document limit
    try:
        doc_limit = int(doc_limit)
    except ValueError:
        doc_limit = 5

    query = query
    query = query.strip()
    orig_query = query
    query = query.split()

    # Exit statements and option setup
    if use_stemming:
        for t in range(len(query)):
            query[t] = porter_stemmer.stem(query[t])

    # Querying the search mode
    try:
        mode = int(mode)
    except ValueError:
        mode = 0

    # Get all accumulated search results for entire query
    query_bulk_t, acc_search_res = 0, [0, {}]
    for term in query:
        search_res, term_bulk_t = test_and_search(term, doc_freq_table, out_folder)
        acc_search_res[0] += search_res[0]
        acc_search_res[1] |= search_res[1]
        query_bulk_t += term_bulk_t
    acc_search_res[1] = list(acc_search_res[1].items())

    # Sort accumulated search results based on search mode
    if mode == 0:
        acc_search_res[1].sort(key=lambda doc: doc[1]['f'], reverse=True)
    elif mode in {1, 2, 3}:
        if mode == 1:
            # BM25
            ranker = BM25Ranker(acc_search_res)
        elif mode == 2:
            # KMeans
            ranker = KMeansRanker(acc_search_res, use_stop_words, use_stemming)
        elif mode == 3:
            # KNN
            ranker = KNNRanker(acc_search_res, use_stop_words, use_stemming)
        doc_rel_scores = ranker.rank(query)
        acc_search_res[1].sort(key=lambda doc: doc_rel_scores.get(doc[0], 0), reverse=True)

    # Results processing
    pretty_print(acc_search_res, orig_query, doc_limit)
    print(f'\n* Query took {query_bulk_t:.3f}s')

    return acc_search_res, orig_query, query_bulk_t
    
if __name__ == "__main__":
    main()
