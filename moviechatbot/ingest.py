import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm

# LangChain + Qdrant imports
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load local .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set.")
if not QDRANT_URL or not QDRANT_API_KEY:
    print("⚠️ Warning: QDRANT_URL or QDRANT_API_KEY not set — local preview only.")

def build_documents(df: pd.DataFrame):
    # Normalize column names for flexible detection
    df.columns = [c.strip() for c in df.columns]

    # Map known IMDb dataset column names
    title_col = next((c for c in ["Series_Title", "Title", "movie", "Movie", "Name"] if c in df.columns), None)
    overview_col = next((c for c in ["Overview", "overview", "plot", "Plot", "Description"] if c in df.columns), None)

    texts, metadatas = [], []

    for _, row in df.iterrows():
        title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else "Unknown Title"
        overview = str(row[overview_col]) if overview_col and pd.notna(row[overview_col]) else ""

        meta = {
            "title": title,
            "released_year": str(row.get("Released_Year", "")),
            "certificate": str(row.get("Certificate", "")),
            "runtime": str(row.get("Runtime", "")),
            "genre": str(row.get("Genre", "")),
            "imdb_rating": str(row.get("IMDB_Rating", "")),
            "meta_score": str(row.get("Meta_score", "")),
            "director": str(row.get("Director", "")),
            "star1": str(row.get("Star1", "")),
            "star2": str(row.get("Star2", "")),
            "star3": str(row.get("Star3", "")),
            "star4": str(row.get("Star4", "")),
            "no_of_votes": str(row.get("No_of_Votes", "")),
            "gross": str(row.get("Gross", "")),
            "poster_link": str(row.get("Poster_Link", "")),
        }

        # Build full content for embedding
        content = f"""Title: {title}
Year: {meta['released_year']}
Certificate: {meta['certificate']}
Runtime: {meta['runtime']}
Genre: {meta['genre']}
IMDb Rating: {meta['imdb_rating']}
Metascore: {meta['meta_score']}
Director: {meta['director']}
Stars: {meta['star1']}, {meta['star2']}, {meta['star3']}, {meta['star4']}
Votes: {meta['no_of_votes']}
Gross: {meta['gross']}

Overview:
{overview}
"""
        texts.append(content)
        metadatas.append(meta)

    return texts, metadatas


def ingest(csv_path: str, collection_name: str = "movie_collection", chunk_size: int = 800, chunk_overlap: int = 100):
    print(f"📖 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    texts, metadatas = build_documents(df)
    print(f"✅ Built {len(texts)} documents. Splitting into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs, doc_metas = [], []

    for text, meta in tqdm(zip(texts, metadatas), total=len(texts)):
        chunks = text_splitter.split_text(text)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            doc_metas.append({**meta, "chunk": i, "source": "imdb_top_1000"})

    print("🚀 Creating embeddings and upserting to Qdrant...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    if not QDRANT_URL or not QDRANT_API_KEY:
        import json
        out = [{"id": i, "text": d, "meta": doc_metas[i]} for i, d in enumerate(docs)]
        with open("local_chunks_preview.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("💾 Saved local_chunks_preview.json (no Qdrant credentials).")
        return

    vectorstore = QdrantVectorStore.from_texts(
        docs,
        embedding=embeddings,
        metadatas=doc_metas,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    print(f"🎉 Inserted {len(docs)} chunks into collection `{collection_name}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to IMDb CSV (e.g. data/imdb_top_1000.csv)")
    parser.add_argument("--collection", default="movie_collection", help="Qdrant collection name")
    args = parser.parse_args()
    ingest(args.csv, args.collection)
