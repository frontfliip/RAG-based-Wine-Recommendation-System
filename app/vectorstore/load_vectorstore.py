from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(path="vectorstore/faiss_index"):
    embedding_fn = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = FAISS.load_local(path, embedding_fn, allow_dangerous_deserialization=True)
    return vectorstore