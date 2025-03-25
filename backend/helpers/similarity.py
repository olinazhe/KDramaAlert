import numpy as np
from typing import List
from collections import defaultdict
# from preprocessing import strip_text

def query_word_counts(query):
    query = query.lower().split(" ")
    query_dictionary = defaultdict(int)
    for word in query:
      query_dictionary[word] += 1
    return query_dictionary

def doc_scores(query_word_counts: dict, inverted_index: dict, idf: dict) -> dict:
    """
    Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    """
    result = defaultdict(float)
    for word in query_word_counts:
      if word in inverted_index and word in idf:
        query_weight = query_word_counts[word] * idf[word]
        for doc, count in inverted_index[word]:
          doc_weight = count * idf[word]
          result[doc] += query_weight * doc_weight
    return result 


def index_search(query: str, inverted_index: dict, idf, doc_norms):
    """
    Search the collection of documents for the given query
    """
    query_tokens = query.lower().split(" ")

    tf= defaultdict(int)
    for token in query_tokens:
        tf[token] += 1

    query_norm = 0
    for word in tf:
      if word in idf:
        word_idf = idf.get(word, 0)
        query_norm += (tf[word] * word_idf) ** 2
    query_norm = np.sqrt(query_norm)

    cossim = doc_scores(tf, inverted_index, idf)

    results = []
    for i in cossim:
      if query_norm > 0 and doc_norms[i] > 0:
        new_score = cossim[i] / (query_norm * doc_norms[i])
        results.append((new_score, i))

    results.sort(key = lambda x : x[0], reverse = True)
    return results

# def get_title_sim(query: str, titles: List[str]) -> np.ndarray:
#     """
#     Returns the similarity between a query and each title using Jaccard similarity in the range [0, 1].
#     """

#     types = set(strip_text(query))
#     result = np.zeros((len(titles),))
#     for i, title in enumerate(titles):
#         title_tokens = set(strip_text(title))
#         result[i] = len(title_tokens.intersection(types)) / len(title_tokens)
#     return result