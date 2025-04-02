import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def create_documents(df):
    all_review_cols = [col for col in df.columns if col.startswith("review_")]
    review_cols = all_review_cols[:5]

    def clean_text(text):
        return str(text).replace("\\n", " ").replace("\n", " ").replace("\r", " ").strip()

    documents = []
    meta_fields = ["price", "points", "province", "variety", "designation", "country", "region_1", "winery"]

    for idx, row in df.iterrows():
        sections = []

        if "title" in df.columns and pd.notna(row["title"]):
            title_clean = clean_text(row["title"])
            sections.append(f"Title: {title_clean}")

        if "description" in df.columns and pd.notna(row["description"]):
            desc_clean = clean_text(row["description"])
            sections.append(f"Description: {desc_clean}")

        meta_parts = []
        for col in meta_fields:
            if col in df.columns and pd.notna(row[col]):
                meta_parts.append(f"{col.capitalize()}: {row[col]}")

        if "vintage" in df.columns and pd.notna(row["vintage"]):
            meta_parts.append(f"Vintage: {row['vintage']}")
        if "wine_color" in df.columns and pd.notna(row["wine_color"]):
            meta_parts.append(f"Wine Color: {row['wine_color']}")

        if meta_parts:
            metadata_section = ", ".join(meta_parts)
            sections.append(metadata_section)

        review_texts = []
        for col in review_cols:
            if pd.notna(row[col]):
                review_clean = clean_text(row[col])
                review_texts.append(review_clean)
        if review_texts:
            reviews_section = "Reviews:\n" + "\n".join(review_texts)
            sections.append(reviews_section)

        meta = {}
        for col in meta_fields + ["id"]:
            if col in df.columns:
                meta[col] = row[col]

        if "vintage" in df.columns:
            meta["vintage"] = row["vintage"]
        if "wine_color" in df.columns:
            meta["wine_color"] = row["wine_color"]

        if "id" in df.columns and pd.notna(row["id"]):
            meta["id"] = row["id"]
        else:
            meta["id"] = f"doc_{idx}"

        full_text = "\n".join(sections)

        doc = Document(page_content=full_text, metadata=meta)
        documents.append(doc)
    return documents


def create_vectorstore(embedding_fn, documents):
    vectorstore = FAISS.from_documents(documents, embedding_fn)
    return vectorstore
