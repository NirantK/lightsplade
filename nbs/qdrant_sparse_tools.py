from typing import Dict

from qdrant_client import models


def convert_sparse_vector(sparse_vector: Dict) -> models.SparseVector:
    indices = []
    values = []

    for idx, value in sparse_vector.items():
        indices.append(int(idx))
        values.append(value)

    return models.SparseVector(indices=indices, values=values)