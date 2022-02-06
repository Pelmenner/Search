import io
import math
import multiprocessing as mp
from collections import Counter
from itertools import islice
from typing import List, Dict

import numpy as np
from tqdm import tqdm

import utils
from article import Article


def initializer(word_vectors_arg, idf_arg, zero_vec_arg):
    global word_vectors, idf, zero_vec
    word_vectors = word_vectors_arg
    idf = idf_arg
    zero_vec = zero_vec_arg


def article_text_to_vector(article: Article) -> np.ndarray:
    return sum(word_vectors.get(w, zero_vec) * idf.get(w, 1) for w in utils.tokenize(article.text))


def article_title_to_vector(article: Article) -> np.ndarray:
    return sum(word_vectors.get(w, zero_vec) * idf.get(w, 1) for w in utils.tokenize(article.title))


class Scorer:
    def __init__(self):
        self.article_vectors = []
        self.title_vectors = []
        self.word_vectors = dict()
        self.idf = dict()
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

    def fit(self, article_list: List[Article], document_index: Dict[str, List[int]]):
        self.word_vectors = self.load_vectors('cc.ru.300.vec', 200000)
        self.zero_vec = np.zeros(shape=self.word_vectors['я'].shape)
        print('training tfidf...')
        self.idf = {key: np.log(len(article_list) / len(document_index[key])) for key in document_index.keys()}
        print('building tf...')
        with mp.Pool() as pool:
            self.text_tf = pool.map(Scorer.count_text_words, article_list)
            self.title_tf = pool.map(Scorer.count_title_words, article_list)

        print('building article vectors')
        with mp.Pool(4, initializer=initializer, initargs=(self.word_vectors, self.idf, self.zero_vec)) as pool:
            self.article_vectors = pool.map(
                article_text_to_vector,
                article_list)
            self.title_vectors = pool.map(
                article_title_to_vector,
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
        return utils.cosine_similarity(article_vec, query_vec)

    def word2vec_title_score(self, keywords: List[str], article_ind: int) -> float:
        query_vec = self.get_query_vec(keywords)
        article_vec = self.title_vectors[article_ind]
        return utils.cosine_similarity(article_vec, query_vec)

    def score(self, query: str, article_ind: int) -> float:
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
