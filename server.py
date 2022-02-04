from flask import Flask, render_template, request
import search
from time import time
from cProfile import Profile

app = Flask(__name__, template_folder='.')
pr = Profile()
pr.disable()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    pr.enable()
    article_indices = search.retrieve_indices(query)
    scored = [(search.articles[article_ind], search.scorer.score(query, article_ind))
              for article_ind in article_indices]

    pr.disable()
    pr.dump_stats('profile.pstat')
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [doc.format(query)+['%.2f' % scr] + [doc.url] for doc, scr in scored]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Pelmen',
        results=results
    )


if __name__ == '__main__':
    search.build_search()
    app.run(debug=True, host='0.0.0.0', port=80)
