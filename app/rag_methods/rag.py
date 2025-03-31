from llm_setup.setup_llm import set_up_llm
from rag_methods.metadata_matching import match_metadata_all
from rag_methods.llm_calls import extract_metadata, get_recommendation
from rag_methods.retrieval_strategies import (
    hyde_retrieval,
    fusion_retrieval,
    hybrid_search_with_metadata,
    naive_retrieval,
)
from rag_methods.clarification import generate_clarifying_questions, filter_answers
from vectorstore.load_vectorstore import load_vectorstore


class RetrievalStrategy:
    NAIVE = 'naive'
    HYBRID = 'hybrid'
    HYDE = 'hyde'
    FUSION = 'fusion'


class RAG:
    def __init__(self, retrieval_strategy: str, k: int = 15):
        self.vectorstore = load_vectorstore()
        self.client = set_up_llm()
        self.retrieval_strategy = retrieval_strategy
        self.k = k
        self.allowed_values = {}  # populate appropriately

    def set_retrieval_strategy(self, strategy: str):
        if strategy not in vars(RetrievalStrategy).values():
            raise ValueError(f"Invalid strategy. Available strategies: {vars(RetrievalStrategy).values()}")
        self.retrieval_strategy = strategy

    def extracted_and_match_metadata(self, query):
        extracted_metadata = extract_metadata(self.client, query)
        matched_metadata = match_metadata_all(extracted_metadata, self.allowed_values)
        return matched_metadata

    def retrieve(self, query: str, matched_metadata):

        if self.retrieval_strategy == RetrievalStrategy.NAIVE:
            return {'naive': naive_retrieval(query, self.vectorstore, k=self.k)}

        elif self.retrieval_strategy == RetrievalStrategy.HYBRID:
            return {'hybrid': hybrid_search_with_metadata(query, self.vectorstore, matched_metadata, k=self.k)}

        elif self.retrieval_strategy == RetrievalStrategy.HYDE:
            return {'hyde': hyde_retrieval(query, self.client, self.vectorstore, matched_metadata, k=self.k)}

        elif self.retrieval_strategy == RetrievalStrategy.FUSION:
            return {'fusion': fusion_retrieval(query, self.client, self.vectorstore, matched_metadata, num_queries=3,
                                               top_k=self.k, dense_k=self.k)}

        else:
            raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")

    def get_final_recommendation(self, retrieval_context, query: str) -> str:
        retrieval_context = "\n\n".join(
            [doc.page_content for strategy_results in retrieval_context.values() for doc in strategy_results]
        )

        recommendation = get_recommendation(self.client, retrieval_context, query)
        return recommendation

    def recommend(self, query):
        matched_metadata = self.extracted_and_match_metadata(query)
        retrieval_context = self.retrieve(query, matched_metadata)
        recommendation = self.get_final_recommendation(retrieval_context, query)
        return recommendation
