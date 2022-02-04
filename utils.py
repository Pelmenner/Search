from typing import List

import nltk

stemmer = nltk.stem.SnowballStemmer(language='russian')


def normalize_word(s: str) -> str:
    return stemmer.stem(s.lower())


def tokenize(text: str) -> List[str]:
    return list(map(normalize_word, nltk.word_tokenize(text.rstrip())))
