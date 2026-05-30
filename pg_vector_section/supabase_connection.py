import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

def connect_to_supabase():
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

    #supabase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings = embeddings,
        collection_name = "production_docs",
        connection = DATABASE_URL,
        use_jsonb = True    #for metadata
    )

    return vectorstore