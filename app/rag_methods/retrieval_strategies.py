from rag_methods.metadata_matching import metadata_matches
from rag_methods.llm_calls import generate_hypothetical_document, generate_queries_llm
from rank_bm25 import BM25Okapi


def metadata_filtering(candidates, constraints, k=15):
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

    filtered = metadata_filtering(
        list(candidate_docs.values()),
        metadata_constraints,
        k=top_k
    )

    fused_docs = sorted(
        filtered,
        key=lambda d: fusion_scores.get(d.metadata.get("id"), 0.0),
        reverse=True
    )
    return fused_docs[:top_k], query_results, fusion_scores


def fusion_retrieval(query, client, vectorstore, metadata, top_k=15, dense_k=15, rrf_k=10, num_queries=3):
    fusion_queries = generate_queries_llm(client, query, num_queries=num_queries)
    fusion_queries.append(query)
    fusion_results, query_results, fusion_scores = reciprocal_rank_fusion(vectorstore, fusion_queries, metadata,
                                                                          top_k=top_k, dense_k=dense_k, rrf_k=rrf_k)
    return fusion_results


def bm25_retrieval(query, documents, k=15):
    tokenized_docs = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in ranked_indices[:k]]


def hybrid_fusion_retrieval(query, vectorstore, documents, bm25_weight=0.5, semantic_weight=0.5, k=50, dense_k=70):
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


def naive_retrieval(query, vectorstore, k=15):
    return vectorstore.similarity_search(query, k=k)
