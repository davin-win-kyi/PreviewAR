from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

def create_embeddings(model_name, docs):
    embeddings = OpenAIEmbeddings(model=model_name)
    vectors = embeddings.embed_documents([d.page_content for d in docs])
    return embeddings, vectors

def create_chroma_db(chunks, embeddings, persist_directory='./chroma_db'):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return db

def reload_chroma_db(embeddings, persist_directory='./chroma_db'):
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return db

def search_db(db, query, k=2):
    results = db.similarity_search(query, k=k)
    return results

def get_rag_db():
    """Load the existing Chroma DB for RAG queries"""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    db = reload_chroma_db(embeddings)
    return db, embeddings

def query_rag_for_dimensions(db, html_content=""):
    """Query RAG system for product dimension selectors"""
    query = 'Amazon product dimensions selectors CSS'
    results = search_db(db, query, k=3)
    return results

def query_rag_for_images(db, html_content=""):
    """Query RAG system for product image selectors"""
    query = 'Amazon product images selectors CSS'
    results = search_db(db, query, k=3)
    return results

def main():
    docs = load_documents("web_scraping_info.md")
    chunks = split_documents(docs)
    embeddings, _ = create_embeddings('text-embedding-3-small', chunks)
    db = create_chroma_db(chunks, embeddings)
    query = 'selectors for product dimensions on Amazon'
    results = search_db(db, query)
    for r in results:
        print(r.page_content)

    # Reload DB without regenerating
    db_reload = reload_chroma_db(embeddings)

if __name__ == "__main__":
    main()


