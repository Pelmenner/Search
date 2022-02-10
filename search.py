import logging
import multiprocessing as mp
import pickle
from cProfile import Profile
from collections import defaultdict
from typing import List
import queue

import pandas as pd

import utils
from article import Article
from scorer import Scorer

RETRIEVE_CNT = 50

index = defaultdict(list)
articles = []
scorer = Scorer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('search')
logger.setLevel(logging.INFO)
pr = Profile()
pr.disable()


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
                result_queue.put(local_index)
                article_queue.task_done()
                return 0

            ind, article = task
            for word in set(utils.tokenize(article.text) + utils.tokenize(article.title)):
                local_index[word].append(ind)
            article_queue.task_done()
        except Exception as e:
            logging.exception(f'{mp.process.current_process()} task failed. {e}')


def load_articles(filename: str = 'habr_posts.csv'):
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

    result_dicts = []
    while True:
        try:
            result = result_queue.get(False, 0.01)
            result_dicts.append(result)
        except queue.Empty:
            pass
        all_exited = True
        for t in processes:
            if t.exitcode is None:
                all_exited = False
                break
        if all_exited & result_queue.empty():
            break

    article_queue.join()
    for process in processes:
        process.join()

    for local_index in result_dicts:
        for key in local_index:
            index[key] += local_index[key]

    for key in index:
        index[key].sort()


def build_search():
    if load_index():
        logger.info('using prebuilt index and scorer')
        return

    pr.enable()

    logger.info('loading articles...')
    load_articles()

    logger.info('building index...')
    build_index()

    logger.info('fitting scorer...')
    scorer.fit(articles, index)

    logger.info('saving...')
    save_index()
    logger.info('index successfully built')

    pr.disable()
    pr.dump_stats('build_search_profile.pstat')


def retrieve_indices(query: str) -> List[int]:
    logger.info(f'query: {query}')
    keywords = utils.tokenize(query)
    if not keywords:
        if query == '':
            return []
        else:
            return list(range(RETRIEVE_CNT))

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

    return s[:RETRIEVE_CNT]
