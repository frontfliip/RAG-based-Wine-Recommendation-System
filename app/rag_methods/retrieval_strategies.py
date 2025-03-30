from rag_methods.metadata_matching import metadata_matches
from rag_methods.llm_calls import generate_hypothetical_document, generate_queries_llm

def hybrid_search_with_metadata(query, vectorstore, constraints, k=15, dense_k=50):
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
    candidates = vectorstore.similarity_search(query, k=dense_k)
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


def hyde_retrieval(query, client, vectorstore, metadata, k=15):
    hypo_doc = generate_hypothetical_document(client, query)
    retrieved_docs = hybrid_search_with_metadata(hypo_doc, vectorstore, metadata, k=k)
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

def naive_retrieval(query, vectorstore,k=15):
    return vectorstore.similarity_search(query, k=k)