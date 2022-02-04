import io
import math
import multiprocessing as mp
import pickle
from collections import Counter, defaultdict
from itertools import islice
from typing import List

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

RETRIEVE_CNT = 50


# TODO: don't save TfidfVectorizer
# TODO: check tfidf correctness
# TODO: remove duplicates
# TODO: refactor...
# TODO: -> github


class Article:
    def __init__(self, title: str, author: str, text: str, popularity: int, rating: int, id: int):
        self.title = title
        self.author = author
        self.text = text
        self.popularity = popularity
        self.rating = rating
        self.id = id

    @property
    def url(self):
        return f'https://habr.com/ru/post/{self.id}/'

    def format(self, query):
        return [self.title, self.text[:50] + ' ...']


def article_text_to_vector(article: Article) -> np.ndarray:
    return sum(scorer.word_vectors.get(w, scorer.zero_vec) * scorer.get_idf(w) for w in tokenize(article.text))


def normalize_word(s: str) -> str:
    return stemmer.stem(s.lower())


def tokenize(text: str) -> List[str]:
    return list(map(normalize_word, nltk.word_tokenize(text.rstrip())))


class Scorer:
    def __init__(self):
        self.article_vectors = []
        self.title_vectors = []
        self.word_vectors = dict()
        self.idf = dict()
        self.tfidf = TfidfVectorizer(tokenizer=tokenize)
        self.zero_vec = np.array([])
        self.text_tf = []
        self.title_tf = []

    @staticmethod
    def load_vectors(filename: str, limit: int):
        fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(islice(fin, limit), total=limit):
            tokens = line.rstrip().split(' ')
            data[normalize_word(tokens[0])] = np.array(list(map(float, tokens[1:])))
        return data

    def text_to_vector(self, text: str) -> np.ndarray:
        return sum(self.word_vectors.get(w, self.zero_vec) * self.get_idf(w) for w in tokenize(text))

    @staticmethod
    def count_text_words(article: Article):
        return Counter(tokenize(article.text))

    @staticmethod
    def count_title_words(article: Article):
        return Counter(tokenize(article.title))

    def fit(self, article_list: List[Article]):
        self.word_vectors = self.load_vectors('cc.ru.300.vec', 200000)
        self.zero_vec = np.zeros(shape=self.word_vectors['я'].shape)
        print('training tfidf...')
        self.tfidf.fit([article.text for article in article_list])
        self.idf = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_ - 1))
        print('building tf...')
        with mp.Pool() as pool:
            self.text_tf = pool.map(Scorer.count_text_words, article_list)
            self.title_tf = pool.map(Scorer.count_title_words, article_list)
            # self.tf = [Counter(tokenize(article.text)) for article in articles]
            # TODO: fix multiprocessing
        print('building article vectors')
        # with mp.Pool() as pool:
        #     pass
        # f = partial(Scorer.article_text_to_vector, self=self)
        # self.article_vectors = pool.map(article_text_to_vector, article_list)
        # TODO: why isn't this working???
        self.article_vectors = [self.text_to_vector(article.text) for article in articles]
        self.title_vectors = [self.text_to_vector(article.title) for article in articles]

    def tfidf_text_score(self, keywords: List[str], article_ind: int) -> float:
        score = sum(
            math.log(1 + self.text_tf[article_ind].get(keyword) / len(articles[article_ind].text)) * self.get_idf(
                keyword)
            for keyword in keywords)
        return score / len(keywords)

    def tfidf_title_score(self, keywords: List[str], article_ind: int) -> float:
        score = sum(
            math.log(1 + self.title_tf[article_ind].get(keyword) / len(articles[article_ind].title)) * self.get_idf(
                keyword)
            for keyword in keywords)
        return score / len(keywords)

    def get_idf(self, word: str) -> float:
        return self.idf.get(word, 1)

    def word2vec_text_score(self, keywords: List[str], article_ind: int) -> float:
        query_vec = sum(self.word_vectors.get(w, self.zero_vec) * self.get_idf(w) for w in keywords)
        article_vec = self.article_vectors[article_ind]
        return query_vec.dot(article_vec) / np.linalg.norm(article_vec) / np.linalg.norm(query_vec)

    def word2vec_title_score(self, keywords: List[str], article_ind: int) -> float:
        query_vec = sum(self.word_vectors.get(w, self.zero_vec) * self.get_idf(w) for w in keywords)
        article_vec = self.title_vectors[article_ind]
        return query_vec.dot(article_vec) / np.linalg.norm(article_vec) / np.linalg.norm(query_vec)

    def score(self, query: str, article_ind: int) -> float:  # TODO: add other scores and regression on top
        keywords = tokenize(query)
        tfidf_text_score = self.tfidf_text_score(keywords, article_ind)
        tfidf_title_score = self.tfidf_title_score(keywords, article_ind)
        word2vec_text_score = self.word2vec_text_score(keywords, article_ind)
        word2vec_title_score = self.word2vec_title_score(keywords, article_ind)
        print('tfidf text score', tfidf_text_score * 500)
        print('tfidf title score', tfidf_title_score * 500)
        print('word2vec text score', word2vec_text_score)
        print('word2vec title score', word2vec_title_score)
        return tfidf_text_score * 500 + word2vec_text_score + tfidf_title_score * 250 + word2vec_title_score * 0.5


index = defaultdict(list)
articles = []
scorer = Scorer()
stemmer = nltk.stem.SnowballStemmer(language='russian')


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


def process_document_index(article: Article, article_index: int, cur_index: dict):
    for word in set(tokenize(article.text)):
        if word not in cur_index:
            cur_index[word] = []
        cur_index[word] += [article_index]


def process_document_indexes(article_queue: mp.Queue, result_queue: mp.Queue):
    local_index = defaultdict(list)
    while True:
        try:
            article = article_queue.get()
            for word in set(tokenize(article.text)):
                local_index[word].append(word)
            if article is None:
                break
        except Exception as e:
            print(mp.process.current_process(), 'task failed.', e)
    result_queue.put(local_index)


def sort_index_item(global_index: dict, values: list, key: str):
    values.sort()
    global_index[key] = values


def load_articles(filename: str = 'habr_posts.csv'):
    articles_df = pd.read_csv(filename)
    articles_df['rating'] = articles_df['rating'].apply(lambda x: x.replace('–', '-'))  # not the same
    global articles
    articles = [Article(str(s.title), str(s.author_nickname), str(s.text), int(s.views_count), int(s.rating), int(s.id))
                for _, s in articles_df.iterrows()]

    articles.sort(key=lambda x: -x.rating)


def build_index():
    num_workers = mp.cpu_count()
    article_queue = mp.Queue()

    for article in articles:
        article_queue.put(article)

    for worker in range(num_workers):
        article_queue.put(None)

    processes = []
    result_queue = mp.Queue()
    for c in range(num_workers):
        p = mp.Process(target=process_document_indexes, args=(article_queue, result_queue))
        p.name = 'worker' + str(c)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print('joined')
    while not result_queue.empty():
        local_index = result_queue.get()
        for key in local_index:
            index[key] += local_index[key]

    print('sorting index')
    for key in index:
        index[key].sort()
    # global index
    # mgr = mp.Manager()
    # index = mgr.dict()
    # with mp.Pool() as pool:
    #     for i, article in enumerate(articles):
    #         pool.apply_async(process_document_index, (article, i, index))
    #     pool.close()
    #     pool.join()
    #
    # print('sorting index...')
    #
    # with mp.Pool() as pool:
    #     for key, values in index.items():
    #         pool.apply_async(sort_index_item, (index, key, values))
    #     pool.close()
    #     pool.join()
    #     # pool.map(process_document_index, enumerate(articles))
    # index = index.copy()

    # for idx, article in enumerate(articles):
    #     for word in set(tokenize(article.text) + tokenize(article.title)):
    #         index[word].append(idx)


def build_search():
    if load_index():
        print('using built index and scorer')
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
    keywords = tokenize(query)
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
