import multiprocessing as mp
import pickle
from collections import defaultdict
from typing import List

import pandas as pd

import utils
from article import Article
from scorer import Scorer

RETRIEVE_CNT = 50

# TODO: don't save TfidfVectorizer
# TODO: remove duplicates
# TODO: refactor...
# TODO: -> github


index = defaultdict(list)
articles = []
scorer = Scorer()


def load_index() -> bool:
    global index, articles, scorer
    try:
        with open('search_dump.pkl', 'rb') as f:
            index, articles, scorer = pickle.load(f)
        return True
    except OSError:
        return False


def save_index():
    with open('search_dump.pkl', 'wb') as f:
        pickle.dump([index, articles, scorer], f)


def process_document_index(article_queue: mp.JoinableQueue, result_queue: mp.Queue):
    local_index = defaultdict(list)
    while True:
        try:
            task = article_queue.get()
            if task is None:
                article_queue.task_done()
                break

            ind, article = task
            for word in set(utils.tokenize(article.text) + utils.tokenize(article.title)):
                local_index[word].append(ind)
            article_queue.task_done()
        except Exception as e:
            print(mp.process.current_process(), 'task failed.', e)
    result_queue.put(local_index)


def load_articles(filename: str = 'habr_posts100.csv'):
    articles_df = pd.read_csv(filename)
    articles_df['rating'] = articles_df['rating'].apply(lambda x: x.replace('–', '-'))  # not the same
    global articles
    articles = [Article(str(s.title), str(s.author_nickname), str(s.text), int(s.views_count), int(s.rating), int(s.id))
                for _, s in articles_df.iterrows()]

    articles.sort(key=lambda x: -x.rating)


def build_index():
    num_workers = mp.cpu_count()
    article_queue = mp.JoinableQueue()

    for task in enumerate(articles):
        article_queue.put(task)

    for worker in range(num_workers):
        article_queue.put(None)

    processes = []
    result_queue = mp.Queue()
    for c in range(num_workers):
        p = mp.Process(target=process_document_index, args=(article_queue, result_queue))
        p.name = 'worker' + str(c)
        processes.append(p)
        p.start()

    article_queue.join()

    print('joining...')
    while not result_queue.empty():
        local_index = result_queue.get()
        for key in local_index:
            index[key] += local_index[key]

    print('sorting index...')
    for key in index:
        index[key].sort()


def build_search():
    if load_index():
        print('using prebuilt index and scorer')
        return

    print('loading articles...')
    load_articles()

    print('building index...')
    build_index()

    print('fitting scorer...')
    scorer.fit(articles)

    print('saving...')
    save_index()
    print('index successfully built')


def retrieve_indices(query: str) -> List[int]:
    print('query:', query)
    keywords = utils.tokenize(query)
    if not keywords:
        return []

    for keyword in keywords:
        if keyword not in index:
            return []

    pointers = [0] * len(keywords)
    s = []
    while len(s) < RETRIEVE_CNT:
        max_i, max_pointer = max(enumerate(pointers), key=lambda x: index[keywords[x[0]]][x[1]])
        max_article = index[keywords[max_i]][max_pointer]
        for i, _ in enumerate(pointers):
            while pointers[i] < len(index[keywords[i]]) and index[keywords[i]][pointers[i]] < max_article:
                pointers[i] += 1

        in_range = True
        for i, pointer in enumerate(pointers):
            if pointer >= len(index[keywords[i]]):
                in_range = False
                break

        if not in_range:
            break

        equal = True
        for i, pointer in list(enumerate(pointers))[1:]:
            if index[keywords[i]][pointer] != index[keywords[i - 1]][pointers[i - 1]]:
                equal = False
                break

        if equal:
            s.append(index[keywords[0]][pointers[0]])
            pointers[0] += 1
            if pointers[0] >= len(index[keywords[0]]):
                break

    return s[:50]
