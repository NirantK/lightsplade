import math
from typing import Dict, List, Tuple

import nltk
from nltk.stem.snowball import SnowballStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

language = "english"
stopwords = nltk.corpus.stopwords.words(language)

AVG_DOC_SIZE = 200

k = 1.2
b = 0.75
stemmer = SnowballStemmer("english")


def stem_words(words: List[str]) -> List[str]:
    """Stems a list of words using NLTK's Snowball stemmer. And removes stopwords.

    Args:
    words (List[str]): List of words to stem.

    Returns:
    List[str]: List of stemmed words.
    """
    return [stemmer.stem(word) for word in words if word not in stopwords]


def reconstruct_bpe(bpe_tokens: List[str]) -> List[str]:
    """Reconstructs words from BPE tokenized input.

    Args:
    bpe_tokens (List[str]): List of BPE tokens.

    Returns:
    List[str]: Reconstructed words from BPE tokens.
    """
    reconstructed = []
    word = ""
    for i, token in enumerate(bpe_tokens):
        if token.startswith("▁"):
            if (
                word
            ):  # If there's an accumulated word, append it before starting a new one
                reconstructed.append(word)
            word = token[1:]  # Start a new word without the '▁'
        else:
            # Handle cases where a new word should start but there's no '▁'
            if (
                i > 0
                and not bpe_tokens[i - 1].startswith("▁")
                and not bpe_tokens[i - 1].endswith(token[0])
            ):
                if word:  # Finish the current word before starting the new one
                    reconstructed.append(word)
                word = token
            else:
                word += token  # Continue building the current word
    if word:  # Append the last word if any
        reconstructed.append(word)
    return reconstructed


def aggregate_weights(
    words: List[str], original_tokens: List[str], weights: List[float]
) -> List[Tuple[str, float]]:
    """Aggregates weights for stemmed words correctly.

    Args:
    words (List[str]): List of stemmed words.
    original_tokens (List[str]): List of original tokens corresponding to weights.
    weights (List[float]): List of weights for each token.

    Returns:
    List[Tuple[str, float]]: List of tuples containing stemmed words and their aggregated weights.
    """
    weight_dict = {}
    for word, weight in zip(words, weights):
        if word in weight_dict:
            weight_dict[word] += weight
        else:
            weight_dict[word] = weight
    return list(weight_dict.items())


def aggregate_weights_idf(
    words: List[str], original_tokens: List[str], weights: List[float]
) -> List[Tuple[str, float]]:
    """Aggregates weights for stemmed words using IDF-inspired logic.

    Args:
        words (List[str]): List of stemmed words.
        original_tokens (List[str]): List of original tokens corresponding to weights.
        weights (List[float]): List of weights for each token.

    Returns:
        List[Tuple[str, float]]: List of tuples containing stemmed words and their adjusted aggregated weights.
    """
    # Step 1: Aggregate initial weights and calculate document frequencies
    weight_dict = {}
    token_dict = {}  # To count unique original tokens for each stemmed word
    for word, token, weight in zip(words, original_tokens, weights):
        if word in weight_dict:
            weight_dict[word] += weight
        else:
            weight_dict[word] = weight

        if word not in token_dict:
            token_dict[word] = set()
        token_dict[word].add(token)

    # Step 2: Calculate IDF scores and adjust weights
    total_unique_tokens = len(set(original_tokens))
    adjusted_weights = {}
    for word, aggregated_weight in weight_dict.items():
        doc_frequency = len(token_dict[word])
        idf_score = math.log((1 + total_unique_tokens) / (1 + doc_frequency)) + 1
        adjusted_weights[word] = aggregated_weight * idf_score

    # Step 3: Optionally rescore weights based on significance
    adjusted_weights = rescore_weights(adjusted_weights)

    return list(adjusted_weights.items())


def rescore_weights(weight_dict: Dict[str, float]) -> Dict[str, float]:
    """Rescores weights based on their rank to emphasize more important terms.

    Args:
        weight_dict (Dict[str, float]): Dictionary of weights keyed by terms.

    Returns:
        Dict[str, float]: Rescored weight dictionary.
    """
    sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
    rescored_weights = {}
    for i, (word, weight) in enumerate(sorted_weights):
        # Decrease weight logarithmically based on rank
        rescored_weights[word] = math.log(4.0 / (i + 1.0) + 1.0) * weight

    return rescored_weights


def aggregate_weights_idf_k_b(
    words: List[str], original_tokens: List[str], weights: List[float]
) -> List[Tuple[str, float]]:
    """Aggregates weights for stemmed words using IDF-inspired logic with k and b parameters to suppress high-frequency tokens.

    Args:
        words (List[str]): List of stemmed words.
        original_tokens (List[str]): List of original tokens corresponding to weights.
        weights (List[float]): List of weights for each token.

    Returns:
        List[Tuple[str, float]]: List of tuples containing stemmed words and their adjusted aggregated weights.
    """
    # Step 1: Aggregate initial weights and calculate term frequencies
    weight_dict = {}
    token_dict = {}  # To count unique original tokens for each stemmed word
    term_freq = {}  # Term frequency of each stemmed word
    for word, token, weight in zip(words, original_tokens, weights):
        if word in weight_dict:
            weight_dict[word] += weight
            term_freq[word] += 1
        else:
            weight_dict[word] = weight
            term_freq[word] = 1

        if word not in token_dict:
            token_dict[word] = set()
        token_dict[word].add(token)

    total_unique_tokens = len(set(original_tokens))
    total_tokens = len(words)  # Total number of stemmed words processed
    AVG_DOC_SIZE = 200  # An assumed average document size for normalization

    # Constants for normalization
    k = 1.2
    b = 0.75

    # Step 2: Calculate IDF scores and adjust weights using k and b
    adjusted_weights = {}
    for word, aggregated_weight in weight_dict.items():
        doc_frequency = len(token_dict[word])
        tf = term_freq[word]
        idf_score = math.log((1 + total_unique_tokens) / (1 + doc_frequency)) + 1
        normalization = (
            tf * (k + 1) / (k * (1 - b + b * total_tokens / AVG_DOC_SIZE) + tf)
        )
        adjusted_weights[word] = aggregated_weight * idf_score * normalization

    # Step 3: Optionally rescore weights based on significance
    adjusted_weights = rescore_weights(adjusted_weights)

    return adjusted_weights
