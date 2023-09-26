import orjson
import gzip
from bs4 import BeautifulSoup
from math import log

class BM25Ranker:
    def __init__(self, search_res):
        self.search_res = search_res
        

    def rank(self, term):
        """
        r is the number of relevant documents containing the query/term
        R is the number of relevant documents for this query/term

        n is the number of documents containing the query/term
        N is the total number of documents in the collection

        f is the frequency of the query/term in the document (term frequency in the document)
        qf is the frequency of the term in the query
        
        dl is the length of the document (document length)
        avgdl is the document average length along the collection

        k1 is a free parameters of the BM25 function
        k2 is a free parameter for BM25 (query case)
        b is a free parameters of the BM25 function

        (R and r are set to zero if there is no relevance information)
	    """
        doc_freq, queried_docs = self.search_res

        r = 0.0
        R = 0.0

        n = doc_freq 
        N = 212651.0
        
        qf = 1.0 # ?

        k1 = 1.2
        k2 = 100.0
        b = 0.75

        res = {}
        doc_lens = {}
        avgdl = float()
        sum_doc_len = float()
        
        
        with open("data/trec_corpus_5000_compiled_reduced.json", "r") as f_corpus:
            corpus_dct = orjson.loads(f_corpus.read())
            for doc_id_strg in corpus_dct:
                doc_info_dct = corpus_dct[doc_id_strg]

                doc_id = int(doc_id_strg)

                doc_text = doc_info_dct["contents"]
                words = doc_text.split()

                if doc_id in doc_lens: 
                    doc_lens[doc_id] += len(words)
                else:
                    doc_lens[doc_id] = len(words)
        
        sum_doc_len = sum(doc_lens.values())

        for doc_idx, (doc_id, doc) in enumerate(queried_docs, start=1):
            
            f = doc['f']
            dl = doc_lens[doc_id]
            avgdl = float(sum_doc_len) / float(doc_lens[doc_id])
            K = k1 * ((1-b)+b*(float(dl)/float(avgdl)))

            # Calculations
            first = log(((r+0.5)/(R-r+0.5))/((n-r+0.5)/(N-n-R+r+0.5)))
            second = ((k1 + 1)*f)/(K+f)
            third = ((k2+1)*qf)/(k2+qf)

            score = float(first * second * third)

            if doc_id in res: 
                res[doc_id] += score
            else:
                res[doc_id] = score

        return res