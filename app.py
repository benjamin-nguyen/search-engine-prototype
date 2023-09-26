from flask import Flask, render_template, request, session
from search import search

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.jinja_env.add_extension('jinja2.ext.i18n')
app.secret_key = "my-secret-key"

@app.route("/")
def home():
    return render_template("search.html")

@app.route("/results", methods=["POST", "GET"])
def searcher():
    if request.method == "POST" :
        query = request.form["query"]
        session["query"] = query

        doc_limit = request.form["doc_limit"]
        session["doc_limit"] = doc_limit

        stopwords_select = request.form.get('stopwords')
        session["stopwords"] = stopwords_select

        stemming_select = request.form.get('stemming')
        session["stemming"] = stemming_select

        model_select = request.form.get('model')
        session["model"] = model_select

        acc_search_res, orig_query, query_bulk_t = search(doc_limit, stopwords_select, stemming_select, str(query), model_select)


        if session["stopwords"] == "y":
            stopwords = "On"
        else:
            stopwords = "Off"

        if session["stemming"] == "y":
            stemming = "On"
        else:
            stemming = "Off"

        if int(session["model"]) == 0:
            model = "TF"
        elif int(session["model"]) == 1:
            model = "BM25"
        elif int(session["model"]) == 2:
            model = "K-Means"
        else:
            model = "KNN"

        query_t = round(query_bulk_t, 3)
        doc_freq, queried_docs = acc_search_res

        result_cnt = 0
        for _ in range(len(queried_docs)):
            result_cnt += 1
            if result_cnt >= int(doc_limit):
                break
                
        return render_template("results.html", query=session["query"], doc_limit=session["doc_limit"], stopwords=stopwords, stemming=stemming, model=model, queried_docs=queried_docs, doc_freq=doc_freq, orig_query=orig_query, result_cnt=result_cnt, query_t=query_t)

if __name__ == '__main__':
	app.run(debug=True)
