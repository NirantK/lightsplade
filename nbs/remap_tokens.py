import string
from typing import Any, Iterable, List, Tuple

import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


language = "english"

stemmer = SnowballStemmer(language)

tokenizer = WordPunctTokenizer()

stop_words = set(stopwords.words(language))
punctuation = set(string.punctuation)

special_tokens = set(["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"])


def reconstruct_bpe(
    bpe_tokens: Iterable[Tuple[int, str]],
) -> List[Tuple[str, List[int]]]:
    result = []
    acc = ""
    acc_idx = []

    for idx, token in bpe_tokens:
        if token in special_tokens:
            continue

        if token.startswith("##"):
            acc += token[2:]
            acc_idx.append(idx)
        else:
            if acc:
                result.append((acc, acc_idx))
                acc = ""
                acc_idx = []
            acc = token
            acc_idx.append(idx)

    if acc:
        result.append((acc, acc_idx))

    return result


def snowball_tokenize(text: str) -> List[str]:
    return tokenizer.tokenize(text.lower())


def filter_list_tokens(tokens: List[str]) -> List[str]:
    result = []
    for token in tokens:
        if token in stop_words or token in punctuation:
            continue
        result.append(token)
    return result


def filter_pair_tokens(tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    result = []
    for token, value in tokens:
        if token in stop_words or token in punctuation:
            continue
        result.append((token, value))
    return result


def stem_list_tokens(tokens: List[str]) -> List[str]:
    result = []
    for token in tokens:
        processed_token = stemmer.stem(token)
        result.append(processed_token)
    return result


def stem_pair_tokens(tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    result = []
    for token, value in tokens:
        processed_token = stemmer.stem(token)
        result.append((processed_token, value))
    return result


def aggregate_weights(
    tokens: List[Tuple[str, List[int]]], weights: List[float]
) -> List[Tuple[str, float]]:
    result = []
    for token, idxs in tokens:
        sum_weight = sum(weights[idx] for idx in idxs)
        result.append((token, sum_weight))
    return result


AVG_DOC_SIZE = 100


k = 1.2
b = 0.75


def calc_tf(tf, doc_size):
    return (k + 1) * tf / (k * (1 - b + b * doc_size / AVG_DOC_SIZE) + tf)