from typing import List

import nltk
import numpy as np
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
stopwords = nltk.corpus.stopwords.words('russian')


def normalize_word(s: str) -> str:
    return morph.parse(s.lower())[0].normal_form


def tokenize(text: str) -> List[str]:
    return [normalize_word(w) for w in nltk.word_tokenize(text.rstrip()) if len(w) > 2 and w not in stopwords]


def cosine_similarity(v1: np.array, v2: np.array):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return v1.dot(v2) / (norm1 * norm2)
