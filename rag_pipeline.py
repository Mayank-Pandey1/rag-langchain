from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import tempfile
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE = """
# LangChain Knowledge Base

## What is LangChain?

LangChain is a framework for building applications powered by large language models (LLMs).
It helps developers create AI applications such as chatbots, question-answering systems,
RAG (Retrieval-Augmented Generation) pipelines, AI agents, and document analyzers.

---

## Core Components of LangChain

### 1. Models
LangChain supports multiple LLM providers including:
- OpenAI
- Google Gemini
- Anthropic
- Ollama
- Hugging Face

Models can be used for:
- Text generation
- Embeddings
- Chat applications

---

### 2. Embeddings

Embeddings convert text into numerical vectors.
These vectors help measure semantic similarity between pieces of text.

Example embedding providers:
- OpenAI Embeddings
- Gemini Embeddings
- HuggingFace Embeddings

Embeddings are commonly used in:
- Semantic search
- Vector databases
- Recommendation systems
- RAG pipelines

---

### 3. Vector Stores

Vector databases store embeddings efficiently.

Popular vector stores:
- ChromaDB
- FAISS
- Pinecone
- Weaviate

Vector stores allow similarity search between user queries and stored documents.

---

### 4. Text Splitters

Large documents are split into smaller chunks before embedding.

RecursiveCharacterTextSplitter is commonly used because it preserves context while splitting.

Important parameters:
- chunk_size
- chunk_overlap

---

### 5. Retrieval-Augmented Generation (RAG)

RAG combines:
1. Retrieval from a knowledge base
2. Generation using an LLM

Steps in RAG:
1. User asks a question
2. Relevant chunks are retrieved from vector DB
3. Retrieved context is sent to the LLM
4. LLM generates the final answer

---

## Example Use Cases

### Chatbots
AI assistants trained on company documents.

### Document Search
Search through PDFs, notes, or websites using semantic similarity.

### AI Agents
Agents that can use tools, APIs, and reasoning.

### Question Answering
Systems that answer questions from custom data sources.

---

## ChromaDB

Chroma is an open-source vector database commonly used with LangChain.

Features:
- Easy setup
- Persistent storage
- Fast similarity search
- Lightweight and beginner-friendly

Example:
```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model
)

LangChain was founded in October 2022 by Harrison Chase (who serves as CEO) and Ankush Gola (who serves as CTO).
"""

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

def create_kb():
    """Creating a vector store from knowledge base"""

    #split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    doc = Document(page_content = KNOWLEDGE_BASE,
                   metadata = {"source": "langchain_knowledge_base.md"}) 
    
    chunks = splitter.split_documents([doc])

    #creating a vector store db
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings_model,
        persist_directory = tempfile.mkdtemp()
    )

    return vector_store

def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
    )

    #RAG prompt tempelate
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:
        {context}

        Question: {question}
        Answer: 

        Make sure to answer in concise manner,
        and if you don't know just say I don't know
        """
    )

    #format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    #RAG_CHAIN
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  #runnablePassThrough - makes sure the query which comes-in doesn't change
        | prompt
        | llm
        | StrOutputParser()
    )

    #Testing RAG chain
    questions = [
        "What is LangChain?",
        "Who created Langchain?",
        "What is Langgraph used for?"
    ]

    print("Basic RAG Demo: \n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()