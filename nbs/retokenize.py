import math
from typing import List, Tuple

import nltk
from nltk.stem.snowball import SnowballStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


language = "english"
AVG_DOC_SIZE = 200

k = 1.2
b = 0.75
nltk.download("punkt")
stemmer = SnowballStemmer("english")


def reconstruct_bpe(bpe_tokens: List[str]) -> List[str]:
    """Reconstructs words from BPE tokenized input.

    Args:
    bpe_tokens (List[str]): List of BPE tokens.

    Returns:
    List[str]: Reconstructed words from BPE tokens.
    """
    reconstructed = []
    word = ""
    for token in bpe_tokens:
        if token.startswith("▁"):
            if word:
                reconstructed.append(word)
            word = token[1:]  # remove the leading '▁' and start a new word
        else:
            word += token
    if word:
        reconstructed.append(word)  # append the last word
    return reconstructed


def stem_words(words: List[str]) -> List[str]:
    """Stems a list of words using NLTK's Snowball stemmer.

    Args:
    words (List[str]): List of words to stem.

    Returns:
    List[str]: List of stemmed words.
    """
    return [stemmer.stem(word) for word in words]


def aggregate_weights(
    words: List[str], original_tokens: List[str], weights: List[float]
) -> List[Tuple[str, float]]:
    """Aggregates weights for stemmed words.

    Args:
    words (List[str]): List of stemmed words.
    original_tokens (List[str]): List of original tokens corresponding to weights.
    weights (List[float]): List of weights for each token.

    Returns:
    List[Tuple[str, float]]: List of tuples containing stemmed words and their aggregated weights.
    """
    weight_dict = {}
    for word, token in zip(words, original_tokens):
        if token.startswith("▁"):
            token = token[1:]  # normalize the token by removing leading '▁'
        if word in weight_dict:
            weight_dict[word] += weights[original_tokens.index(token)]
        else:
            weight_dict[word] = weights[original_tokens.index(token)]
    return list(weight_dict.items())

def process_text(
    bpe_tokens: List[str], weights: List[float], aggregate_fn: object
) -> List[Tuple[str, float]]:
    """process the entire pipeline of BPE tokenization, stemming, and aggregation.

    Args:
    bpe_tokens (List[str]): BPE tokenized text.
    weights (List[float]): Weights associated with each BPE token.

    Returns:
    List[Tuple[str, float]]: Weighted, stemmed words.
    """
    # Step 1: Reconstruct words from BPE tokens
    reconstructed_words = reconstruct_bpe(bpe_tokens)

    # Step 2: Stem the reconstructed words
    stemmed_words = stem_words(reconstructed_words)

    # Step 3: Aggregate weights based on stemmed words
    aggregated_weights = aggregate_fn(stemmed_words, bpe_tokens, weights)

    return aggregated_weights


# def rescore_vector(vector: dict) -> dict:
#     sorted_vector = sorted(vector.items(), key=lambda x: x[1], reverse=True)

#     new_vector = {}

#     for num, (token, value) in enumerate(sorted_vector):
#         new_vector[token] = math.log(4.0 / (num + 1.0) + 1.0)  # value

#     return new_vector

# def calc_tf(tf, doc_size):
#     return (k + 1) * tf / (k * (1 - b + b * doc_size / AVG_DOC_SIZE) + tf)
