from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain.storage import InMemoryStore # Removed for direct Chroma inspection
from langchain_core.documents import Document
import os

persist_directory = './textbook_test_example1/'
id_key = "doc_id"

print(f"\n🔍 Verifying ChromaDB contents from: {persist_directory}")

if os.path.exists(persist_directory):
    try:
        # Initialize Chroma vectorstore for summaries
        vectorstore_summaries = Chroma(collection_name="summaries", persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())
        # Initialize Chroma vectorstore for original texts
        vectorstore_original_texts = Chroma(collection_name="original_texts", persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())

        print(f"Total summary documents in ChromaDB: {len(vectorstore_summaries.get(include=[])['ids'])}")
        print(f"Total original text documents in ChromaDB: {len(vectorstore_original_texts.get(include=[])['ids'])}")

        # Retrieve and print all summaries
        all_summary_docs = vectorstore_summaries.get(include=['documents', 'metadatas'])
        summary_contents = all_summary_docs['documents']

        if len(summary_contents) > 0:
            print("\n--- All Retrieved Summaries ---")
            for i, content in enumerate(summary_contents):
                print(f"\n--- Summary Document {i+1} ---")
                print(content[:500] + ("..." if len(content) > 500 else "")) # Print first 500 chars or less
                if all_summary_docs['metadatas'] and all_summary_docs['metadatas'][i]:
                    print(f"Metadata: {all_summary_docs['metadatas'][i]}")

        else:
            print("No summary documents found in the database.")

        # Retrieve and print all original texts (if any)
        all_original_docs = vectorstore_original_texts.get(include=['documents', 'metadatas'])
        original_contents = all_original_docs['documents']

        if len(original_contents) > 0:
            print("\n--- All Retrieved Original Texts ---")
            for i, content in enumerate(original_contents):
                print(f"\n--- Original Document {i+1} ---")
                print(content[:500] + ("..." if len(content) > 500 else "")) # Print first 500 chars or less
                if all_original_docs['metadatas'] and all_original_docs['metadatas'][i]:
                    print(f"Metadata: {all_original_docs['metadatas'][i]}")
        else:
            print("No original text documents found in the database.")

    except Exception as e:
        print(f"❌ An error occurred while loading or inspecting ChromaDB: {e}")
else:
    print(f"🚫 ChromaDB directory not found at {persist_directory}. Please ensure the ingestion process has run successfully and created this directory.")

print("--- End of Inspection --- ")
