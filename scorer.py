import io
import math
import multiprocessing as mp
from collections import Counter
from functools import partial
from itertools import islice
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import utils
from article import Article


def article_text_to_vector(article: Article, word_vectors: dict, idf: dict, zero_vec) -> np.ndarray:
    return sum(word_vectors.get(w, zero_vec) * idf.get(w, 1) for w in utils.tokenize(article.text))


def article_title_to_vector(article: Article, word_vectors: dict, idf: dict, zero_vec) -> np.ndarray:
    return sum(word_vectors.get(w, zero_vec) * idf.get(w, 1) for w in utils.tokenize(article.title))


class Scorer:
    def __init__(self):
        self.article_vectors = []
        self.title_vectors = []
        self.word_vectors = dict()
        self.idf = dict()
        self.tfidf = TfidfVectorizer(tokenizer=utils.tokenize)
        self.zero_vec = np.array([])
        self.text_tf = []
        self.title_tf = []
        self.article_lengths = []
        self.title_lengths = []

    @staticmethod
    def load_vectors(filename: str, limit: int):
        fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(islice(fin, limit), total=limit):
            tokens = line.rstrip().split(' ')
            data[utils.normalize_word(tokens[0])] = np.array(list(map(float, tokens[1:])))
        return data

    def text_to_vector(self, text: str) -> np.ndarray:
        return sum(self.word_vectors.get(w, self.zero_vec) * self.get_idf(w) for w in utils.tokenize(text))

    @staticmethod
    def count_text_words(article: Article):
        return Counter(utils.tokenize(article.text))

    @staticmethod
    def count_title_words(article: Article):
        return Counter(utils.tokenize(article.title))

    def fit(self, article_list: List[Article]):
        self.word_vectors = self.load_vectors('cc.ru.300.vec', 100000)
        self.zero_vec = np.zeros(shape=self.word_vectors['я'].shape)
        print('training tfidf...')
        self.tfidf.fit([article.text for article in article_list])
        self.idf = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_ - 1))
        print('building tf...')
        with mp.Pool() as pool:
            self.text_tf = pool.map(Scorer.count_text_words, article_list)
            self.title_tf = pool.map(Scorer.count_title_words, article_list)

        print('building article vectors')
        mgr = mp.Manager()
        word_vectors = mgr.dict(self.word_vectors)
        idf = mgr.dict(self.idf)
        with mp.Pool() as pool:
            self.article_vectors = pool.map(
                partial(article_text_to_vector, word_vectors=word_vectors, idf=idf, zero_vec=self.zero_vec),
                article_list)
            self.title_vectors = pool.map(
                partial(article_title_to_vector, word_vectors=word_vectors, idf=idf, zero_vec=self.zero_vec),
                article_list)

        self.article_lengths = [len(article.text) for article in article_list]
        self.title_lengths = [len(article.title) for article in article_list]

    def tfidf_text_score(self, keywords: List[str], article_ind: int) -> float:
        score = sum(
            math.log(1 + self.text_tf[article_ind].get(keyword, 0) / self.article_lengths[article_ind]) * self.get_idf(
                keyword)
            for keyword in keywords)
        return score / len(keywords)

    def tfidf_title_score(self, keywords: List[str], article_ind: int) -> float:
        score = sum(
            math.log(1 + self.title_tf[article_ind].get(keyword, 0) / self.title_lengths[article_ind]) * self.get_idf(
                keyword)
            for keyword in keywords)
        return score / len(keywords)

    def get_idf(self, word: str) -> float:
        return self.idf.get(word, 1)

    def get_query_vec(self, keywords: List[str]) -> np.ndarray:
        return sum(self.word_vectors.get(w, self.zero_vec) * self.get_idf(w) for w in keywords)

    def word2vec_text_score(self, keywords: List[str], article_ind: int) -> float:
        query_vec = self.get_query_vec(keywords)
        article_vec = self.article_vectors[article_ind]
        return query_vec.dot(article_vec) / np.linalg.norm(article_vec) / np.linalg.norm(query_vec)

    def word2vec_title_score(self, keywords: List[str], article_ind: int) -> float:
        query_vec = self.get_query_vec(keywords)
        article_vec = self.title_vectors[article_ind]
        return query_vec.dot(article_vec) / np.linalg.norm(article_vec) / np.linalg.norm(query_vec)

    def score(self, query: str, article_ind: int) -> float:  # TODO: add regression on top
        keywords = utils.tokenize(query)
        tfidf_text_score = self.tfidf_text_score(keywords, article_ind)
        tfidf_title_score = self.tfidf_title_score(keywords, article_ind)
        word2vec_text_score = self.word2vec_text_score(keywords, article_ind)
        word2vec_title_score = self.word2vec_title_score(keywords, article_ind)
        # print('tfidf text score', tfidf_text_score * 500)
        # print('tfidf title score', tfidf_title_score * 50)
        # print('word2vec text score', word2vec_text_score)
        # print('word2vec title score', word2vec_title_score)
        return tfidf_text_score * 500 + word2vec_text_score + tfidf_title_score * 50 + word2vec_title_score * 0.5
