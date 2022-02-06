from typing import List

import nltk
import numpy as np

stemmer = nltk.stem.SnowballStemmer(language='russian')


def normalize_word(s: str) -> str:
    return stemmer.stem(s.lower())


def tokenize(text: str) -> List[str]:
    return list(map(normalize_word, nltk.word_tokenize(text.rstrip())))


def cosine_similarity(v1: np.array, v2: np.array):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return v1.dot(v2) / (norm1 * norm2)
