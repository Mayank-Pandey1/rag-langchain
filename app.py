import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(collection_name)

documents = [
    {"id": "doc3", "text": "Hello, World!", "metadata": {"chapter": 3, "verse": 16}},
    {"id": "doc4", "text": "Hello, Mayank", "metadata": {"chapter": 3, "verse": 15}},
    {"id": "doc5", "text": "Good to see you!", "metadata": {"chapter": 3, "verse": 5}}
]

for doc in documents:
    collection.upsert(
        ids=doc["id"],
        documents=doc["text"],
        metadatas=doc["metadata"]
    )

#define a query text
query_text = "Hello, World"

results = collection.query(
    query_texts=[query_text],
    n_results=3
)
print(results)