#!/usr/bin/env python3
"""
Debug script for UnifiedRetriever retrieval issues
"""
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
import os

def debug_collections():
    print("=== ChromaDB Collections Debug ===")
    
    persist_directory = '../../chroma/Ears'
    id_key = "doc_id"
    
    open_clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    
    # Vectorstore for summaries (for similarity search)
    vectorstore = Chroma(
        collection_name="summaries_and_images",
        persist_directory=persist_directory,
        embedding_function=open_clip_embeddings
    )
    
    # Persistent docstore for originals (all modalities)
    docstore = Chroma(
        collection_name="originals",
        persist_directory=persist_directory,
        embedding_function=open_clip_embeddings
    )
    
    # Debug: Check collection access
    print(f"Vectorstore collection: {vectorstore._collection}")
    print(f"Docstore collection: {docstore._collection}")
    print(f"Vectorstore collection name: {vectorstore._collection.name}")
    print(f"Docstore collection name: {docstore._collection.name}")
    
    # Test a simple query
    test_query = "ear infection"
    print(f"\n=== Testing query: '{test_query}' ===")
    
    # Get results from vectorstore
    results = vectorstore.similarity_search_with_score(test_query, k=3)
    print(f"Found {len(results)} results from vectorstore")
    
    for i, (doc, score) in enumerate(results):
        doc_id = doc.metadata.get(id_key)
        print(f"\nResult {i}:")
        print(f"  Doc ID: {doc_id}")
        print(f"  Score: {score}")
        print(f"  Metadata: {doc.metadata}")
        print(f"  Content preview: {doc.page_content[:100]}...")
        
        # Try to get original from docstore
        print(f"  Trying to fetch original from docstore...")
        try:
            # Method 1: Direct collection access
            original = docstore._collection.get(ids=[doc_id], include=["documents", "metadatas"])
            print(f"  Collection.get() result: {original}")
            
            if original and original.get("documents"):
                print(f"  Original document found: {len(original['documents'])} docs")
                if original["documents"][0]:
                    print(f"  First doc preview: {str(original['documents'][0])[:100]}...")
                else:
                    print(f"  First doc is None/empty")
            else:
                print(f"  No documents found in original collection")
                
        except Exception as e:
            print(f"  Error accessing collection: {e}")
            
        # Method 2: Try alternative approach
        try:
            # Check if the document exists in docstore at all
            docstore_docs = docstore.get(ids=[doc_id])
            print(f"  Docstore.get() result: {docstore_docs}")
        except Exception as e:
            print(f"  Error with docstore.get(): {e}")
    
    # Debug: List some IDs from each collection
    print(f"\n=== Collection Contents Debug ===")
    try:
        vectorstore_peek = vectorstore._collection.peek(limit=5)
        print(f"Vectorstore peek (first 5): {vectorstore_peek}")
    except Exception as e:
        print(f"Error peeking vectorstore: {e}")
    
    try:
        docstore_peek = docstore._collection.peek(limit=5)
        print(f"Docstore peek (first 5): {docstore_peek}")
    except Exception as e:
        print(f"Error peeking docstore: {e}")

if __name__ == "__main__":
    debug_collections()