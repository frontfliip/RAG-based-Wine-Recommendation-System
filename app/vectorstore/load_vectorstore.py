from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

def load_vectorstore(embedding="openai"):
    embedding = embedding.lower()
    config = {
        "mpnet": {
            "fn": HuggingFaceEmbeddings(model_name="all-mpnet-base-v2"),
            "default_path": "app/vectorstore/faiss_index_all_mpnet_base_v2",
        },
        "roberta": {
            "fn": HuggingFaceEmbeddings(model_name="all-roberta-large-v1"),
            "default_path": "app/vectorstore/faiss_index_all_roberta_large_v1",
        },
        "openai": {
            "fn": OpenAIEmbeddings(model="text-embedding-3-large"),
            "default_path": "app/vectorstore/faiss_index_text_embedding_3_large",
        },
    }

    if embedding not in config:
        valid = ", ".join(config.keys())
        raise ValueError(f"Unknown embedding '{embedding}'. Valid options: {valid}")

    embedding_fn = config[embedding]["fn"]
    index_path = config[embedding]["default_path"]

    vectorstore = FAISS.load_local(index_path, embedding_fn, allow_dangerous_deserialization=True)
    return vectorstore
