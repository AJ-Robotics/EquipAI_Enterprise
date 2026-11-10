from rag_engine import VectorStore
import os

def test_retrieval():
    if not os.path.exists("vector_store.pkl"):
        print("FAIL: Vector store file not found.")
        return

    store = VectorStore()
    store.load("vector_store.pkl")
    
    # Test 1: Content check
    if len(store.metadata) == 0:
        print("FAIL: Vector store is empty.")
        return
    print(f"PASS: Vector store has {len(store.metadata)} chunks.")
    
    # Test 2: Retrieval
    query = "What is the capital of France?"
    results = store.search(query, k=1)
    
    if not results:
        print("FAIL: No results returned.")
        return
        
    top_text = results[0]['metadata']['chunk_text']
    print(f"Query: {query}")
    print(f"Retrieved: {top_text}")
    print(f"Score: {results[0]['score']}")
    
    if "Paris" in top_text:
        print("PASS: Retrieved correct context.")
    else:
        print("FAIL: Did not retrieve correct context.")

if __name__ == "__main__":
    test_retrieval()
