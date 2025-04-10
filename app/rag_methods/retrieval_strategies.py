from rag_methods.metadata_matching import metadata_matches
from rag_methods.llm_calls import generate_hypothetical_document, generate_queries_llm
from rank_bm25 import BM25Okapi


def metadata_filtering(candidates, constraints, k=15):
    """
    Performs a hybrid search using a pre-built FAISS vector store that includes metadata.
    Gradually relaxes constraints in an iterative order (by groups) until at least k documents are matched.

    The relaxation groups are defined such that:
      - "min_price" and "max_price" are always relaxed together.
      - "min_vintage" and "max_vintage" are always relaxed together.
      - Other fields (e.g., "points", "country", "province") are relaxed in separate groups.

    If after relaxing all groups there are fewer than k results, the remainder is filled with raw dense search candidates.

    Args:
        vectorstore: A FAISS vector store built from your Document objects.
        query (str): The user's query.
        constraints (dict): Normalized metadata constraints, e.g.:
            {
              "min_price": 17.0,
              "max_price": 17.0,
              "points": 89,
              "variety_designation": "Rosé",
              "country": "US",
              "province": "New York",
              "wine_color": "Rosé",
              "min_vintage": 2000,
              "max_vintage": 2020
            }
        k (int): Number of final documents to return.
        dense_k (int): Number of documents to retrieve initially.

    Returns:
        List[Document]: A list of up to k Document objects.
    """
    results = []
    seen_ids = set()

    relaxation_groups = [
        ["min_price", "max_price"],
        ["points"],
        ["min_vintage", "max_vintage"],
        ["country"],
        ["province"]
    ]

    current_constraints = constraints.copy()
    group_index = 0

    while group_index <= len(relaxation_groups) and len(results) < k:
        for doc in candidates:
            doc_id = doc.metadata.get("id")
            if doc_id in seen_ids:
                continue
            if metadata_matches(doc.metadata, current_constraints):
                results.append(doc)
                seen_ids.add(doc_id)
        if len(results) >= k:
            break
        if group_index < len(relaxation_groups):
            for key in relaxation_groups[group_index]:
                current_constraints[key] = "-"
        group_index += 1

    if len(results) < k:
        desired_color = constraints.get("wine_color", "-").strip().lower()
        for doc in candidates:
            doc_id = doc.metadata.get("id")
            if doc_id not in seen_ids:
                doc_color = doc.metadata.get("wine_color", "").strip().lower()
                if desired_color != "-" and doc_color != desired_color:
                    continue
                results.append(doc)
                seen_ids.add(doc_id)
            if len(results) >= k:
                break

    return results[:k]


def hyde_retrieval(query, client, vectorstore, metadata, k=15, dense_k=50):
    hypo_doc = generate_hypothetical_document(client, query)
    candidates = vectorstore.similarity_search(hypo_doc, k=dense_k)
    retrieved_docs = metadata_filtering(candidates, metadata, k=k)
    return retrieved_docs


def reciprocal_rank_fusion(vectorstore, queries, metadata_constraints, top_k=10, dense_k=10, rrf_k=10):
    """
    Performs reciprocal rank fusion (RRF) over multiple query variations using a single set of metadata constraints.
    It uses the vectorstore's built-in similarity_search method to retrieve candidate documents for each query,
    then applies the metadata filtering (using the same constraints for all queries) to the combined candidates.

    Args:
      vectorstore: A FAISS vectorstore built via FAISS.from_documents, with a similarity_search(query, k) method.
      queries (List[str]): A list of query variations.
      metadata_constraints (dict): Metadata constraints extracted from the original query.
      top_k (int): The number of final documents to return.
      dense_k (int): The number of documents to retrieve per query.
      rrf_k (int): Smoothing constant for reciprocal rank fusion (default is 10).

    Returns:
      fused_docs (List[Document]): The final list of top documents after fusion and metadata filtering.
      query_results (dict): A mapping from each query to its raw candidate results.
      fusion_scores (dict): A dictionary mapping document identifiers to their cumulative fusion scores.
    """
    fusion_scores = {}
    query_results = {}
    candidate_docs = {}

    for query in queries:
        results = vectorstore.similarity_search(query, k=dense_k)
        query_results[query] = results
        for rank, doc in enumerate(results):
            score = 1.0 / (rank + rrf_k)
            doc_id = doc.metadata.get("id")
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + score
            candidate_docs[doc_id] = doc

    filtered_candidates = {
        doc_id: doc
        for doc_id, doc in candidate_docs.items()
        if metadata_matches(doc.metadata, metadata_constraints)
    }

    fused_docs = sorted(filtered_candidates.values(),
                        key=lambda d: fusion_scores.get(d.metadata.get("id"), 0),
                        reverse=True)

    return fused_docs[:top_k], query_results, fusion_scores

def fusion_retrieval(query, client, vectorstore, metadata, top_k=15, dense_k=15, rrf_k=10, num_queries=3):
    fusion_queries = generate_queries_llm(client, query, num_queries=num_queries)
    fusion_queries.append(query)
    fusion_results, query_results, fusion_scores = reciprocal_rank_fusion(vectorstore, fusion_queries, metadata, top_k=top_k, dense_k=dense_k, rrf_k=rrf_k)
    return fusion_results

def bm25_retrieval(query, documents, k=15):
    """
    Performs BM25 based retrieval on a list of Document objects using the 'page_content' attribute.

    Args:
        query (str): The user's query string.
        documents (List[Document]): A list of Document objects that have a 'page_content' attribute.
        k (int): The number of top documents to return.

    Returns:
        List[Document]: A list of the top-k Document objects ranked by BM25 scores.
    """
    tokenized_docs = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked_indices[:k]]


def hybrid_fusion_retrieval(query, vectorstore, documents, bm25_weight=0.5, semantic_weight=0.5, k=50, dense_k=70):
    """
    Performs a hybrid retrieval by combining semantic (dense) search with keyword-based BM25 search.

    This function retrieves candidate documents using both methods:
      - Dense retrieval using the vectorstore's similarity_search method.
      - Sparse retrieval using BM25 (via bm25_retrieval) on the full documents list.

    The results of each method are fused using a reciprocal rank fusion-inspired approach, where a document's score is computed as:
      score = (weight / (rank + 1))
    with separate weights for BM25 and dense results. The documents are then ranked by the combined score.

    Args:
        query (str): The user's query string.
        vectorstore: A FAISS vector store that provides similarity_search(query, k) for dense retrieval.
        documents (List[Document]): A list of Document objects for BM25 sparse retrieval.
        bm25_weight (float): Weight for the BM25 component of the score.
        semantic_weight (float): Weight for the dense retrieval component of the score.
        k (int): Number of final documents to return.
        dense_k (int): Number of candidate documents to retrieve from each method.

    Returns:
        List[Document]: A list of the top-k Document objects based on combined scores.
    """
    dense_results = vectorstore.similarity_search(query, k=dense_k)

    bm25_results = bm25_retrieval(query, documents, k=dense_k)
    fusion_scores = {}
    candidate_docs = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc.metadata.get("id")
        score = semantic_weight / (rank + 1)
        fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + score
        candidate_docs[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("id")
        score = bm25_weight / (rank + 1)
        fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + score
        candidate_docs[doc_id] = doc

    ranked_doc_ids = sorted(fusion_scores, key=lambda doc_id: fusion_scores[doc_id], reverse=True)
    return [candidate_docs[doc_id] for doc_id in ranked_doc_ids[:k]]


def hybrid_retrieval(query, vectorstore, documents, metadata, bm25_weight=0.5, semantic_weight=0.5, k=15, dense_k=50):
    ranked_documents = hybrid_fusion_retrieval(query, vectorstore, documents, k=dense_k, dense_k=dense_k + 25,
                                               bm25_weight=bm25_weight, semantic_weight=semantic_weight)
    filtered_documents = metadata_filtering(ranked_documents, metadata, k=k)
    return filtered_documents


def naive_retrieval(query, vectorstore,k=15):
    return vectorstore.similarity_search(query, k=k)