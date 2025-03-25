import numpy as np
from typing import List

def get_sim(q, term_doc_matrix, tfidf):
    """Returns cosine similarity
    """
    similarities = []
    for d in term_doc_matrix:
        similarities.append(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)))
    return similarities
