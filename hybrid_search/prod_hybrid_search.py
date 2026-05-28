from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001"
)

documents = [
    Document(
    page_content='Product SKU-7742X is our flagship router. It supports '
                 'gigabit speeds and advanced QoS features.',
    metadata={'type': 'product'}
    ),

    Document(
        page_content='For network connectivity issues, first check the '
                    'ethernet cable and router status lights.',
        metadata={'type': 'troubleshooting'}
    ),

    Document(
        page_content='Error code E_CONN_REFUSED indicates the server '
                     'rejected the connection. Check firewall settings.',
        metadata={'type': 'error'}
    ),

    Document(
        page_content='The authentication process requires valid credentials. '
                     'Use OAuth2 for secure API access.',
        metadata={'type': 'auth'}
    ),
]

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name = "hybrid-test"
)

vector_retriever = vector_store.as_retriever(search_kwargs = {'k': 3})
print("Vector retriever ready")


#BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,       
)
bm25_retriever.k = 3  #return top 3


ensemble_retriever = EnsembleRetriever(
    retrievers = [vector_retriever, bm25_retriever],
    weights = [0.5, 0.5]
)

print("Hybrid retriever ready")


def test_query(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f'\n{name} - Query: \"{query}\"')

    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"{i+1}. {preview}")
    
    return results


# Test queries designed to challenge vector search
test_queries = [
    'SKU-7742X specifications',    # Exact product code
    'E_CONN_REFUSED error',        # Error code
    'How do I authenticate?',      # Semantic question
    'WCAG compliance',             # Acronym
    'router configuration',        # General semantic
]

for query in test_queries:
    vector_results = test_query(query, 'VECTOR', vector_retriever)

    bm25_results = test_query(query, 'BM25', bm25_retriever)

    hybrid_results = test_query(query, 'HYBRID', ensemble_retriever)