from llm_setup.setup_llm import set_up_llm
from rag_methods.metadata_matching import match_metadata_all, get_allowed_values, get_similar_wine
from rag_methods.llm_calls import extract_metadata, get_recommendation, rewrite_query_remove_negative_metadata, classify_query_intent
from rag_methods.retrieval_strategies import (
    hyde_retrieval,
    fusion_retrieval,
    hybrid_search_with_metadata,
    naive_retrieval,
)
from vectorstore.load_vectorstore import load_vectorstore

# from vectorstore.create_vectorstore import create_vectorstore, create_documents
# from langchain_community.embeddings import HuggingFaceEmbeddings



class RetrievalStrategy:
    NAIVE = 'naive'
    HYBRID = 'hybrid'
    HYDE = 'hyde'
    FUSION = 'fusion'


class RAG:
    def __init__(self, df, retrieval_strategy: str, k: int = 15):
        self.vectorstore = load_vectorstore()

        # embedding_fn = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        # documents = create_documents(df)
        # self.vectorstore = create_vectorstore(embedding_fn, documents)

        self.client = set_up_llm()
        self.retrieval_strategy = retrieval_strategy
        self.k = k
        self.allowed_values = get_allowed_values(df)
        self.df = df

    def set_retrieval_strategy(self, strategy: str):
        if strategy not in vars(RetrievalStrategy).values():
            raise ValueError(f"Invalid strategy. Available strategies: {vars(RetrievalStrategy).values()}")
        self.retrieval_strategy = strategy

    def extracted_and_match_metadata(self, query):
        extracted_metadata = extract_metadata(self.client, query)
        matched_metadata = match_metadata_all(extracted_metadata, self.allowed_values)
        return extracted_metadata, matched_metadata

    def retrieve(self, query: str, matched_metadata, similar_intent=False):
        k = self.k + 1
        if similar_intent:
            k += 1

        if self.retrieval_strategy == RetrievalStrategy.NAIVE:
            return {'naive': naive_retrieval(query, self.vectorstore, k=k)}

        elif self.retrieval_strategy == RetrievalStrategy.HYBRID:
            return {'hybrid': hybrid_search_with_metadata(query, self.vectorstore, matched_metadata, k=k)}

        elif self.retrieval_strategy == RetrievalStrategy.HYDE:
            return {'hyde': hyde_retrieval(query, self.client, self.vectorstore, matched_metadata, k=k)}

        elif self.retrieval_strategy == RetrievalStrategy.FUSION:
            return {'fusion': fusion_retrieval(query, self.client, self.vectorstore, matched_metadata, num_queries=3,
                                               top_k=k, dense_k=k)}
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")

    def get_final_recommendation(self, retrieval_context, query: str, reference_doc=None, reference_wine_present=False) -> str:
        retrieval_context = "\n\n".join(
            [doc.page_content for strategy_results in retrieval_context.values() for doc in strategy_results]
        )

        recommendation = get_recommendation(self.client, retrieval_context, query, reference_doc, reference_wine_present)
        return recommendation

    def recommend(self, query):
        def filter_reference_doc(result, reference_doc):
            ref_id = reference_doc.metadata.get("id")
            return [doc for doc in result if doc.metadata.get("id") != ref_id]

        query_intent = classify_query_intent(self.client, query)

        extracted_metadata, matched_metadata = self.extracted_and_match_metadata(query)
        rewritten_query = rewrite_query_remove_negative_metadata(self.client, query, extracted_metadata['negative'])
        retrieval_context = self.retrieve(rewritten_query, matched_metadata)
        if query_intent['intent'] == 'normal':
            recommendation = self.get_final_recommendation(retrieval_context, query)
            return recommendation
        if query_intent['intent'] == 'similar':
            similar_wine = get_similar_wine(self.df, query_intent['reference'])
            retrieval_context[self.retrieval_strategy] = filter_reference_doc(retrieval_context[self.retrieval_strategy], similar_wine)
            recommendation = self.get_final_recommendation(retrieval_context, query, reference_doc=similar_wine, reference_wine_present=True)
            return recommendation


