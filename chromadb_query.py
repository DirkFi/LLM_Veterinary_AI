import os
import base64
from PIL import Image

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def query_chromadb(query_text: str, persist_directory: str, collection_name: str, k: int = 5):
    """
    Queries a specified ChromaDB collection for documents similar to the query text
    and displays them with their similarity scores.

    Args:
        query_text (str): The text query to search for.
        persist_directory (str): The directory where the ChromaDB collection is persisted.
        collection_name (str): The name of the collection to query.
        k (int): The number of top relevant documents to retrieve. Defaults to 5.

    Returns:
        list: A list of (Document, score) tuples retrieved from the ChromaDB.
    """
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=GPT4AllEmbeddings(),
            persist_directory=persist_directory
        )

        print(f"\n🔍 Querying ChromaDB collection '{collection_name}' for: '{query_text}'")
        
        # Perform similarity search with scores
        results_with_scores = vectorstore.similarity_search_with_score(query_text, k=k)
        
        if results_with_scores:
            print(f"Found {len(results_with_scores)} relevant documents in '{collection_name}':")
            for i, (doc, score) in enumerate(results_with_scores):
                print(f"--- Document {i+1} (from {collection_name}) - Score: {score:.4f} ---")
                print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                if doc.metadata:
                    print(f"Metadata: {doc.metadata}")
        else:
            print(f"No relevant documents found in '{collection_name}' for the query.")
        
        return results_with_scores

    except Exception as e:
        print(f"❌ An error occurred while querying ChromaDB: {e}")
        return []

def get_doc_by_id(doc_id: str, persist_directory: str, collection_name: str) -> Document | None:
    """
    Retrieves a document directly by its ID from a specified ChromaDB collection.
    """
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=GPT4AllEmbeddings(),
            persist_directory=persist_directory
        )
        result = vectorstore.get(ids=[doc_id], include=['documents', 'metadatas'])
        print(f"DEBUG (get_doc_by_id): Raw result for ID '{doc_id}': {result}") # Debugging line
        if result and result['documents']:
            return Document(page_content=result['documents'][0], metadata=result['metadatas'][0])
        return None
    except Exception as e:
        print(f"❌ Error retrieving document by ID {doc_id}: {e}")
        return None

def generate_answer_from_docs(query: str, documents: list, model_name: str = "llama3.2") -> str:
    """
    Generates a single, consolidated answer to a query based on a list of documents.

    Args:
        query (str): The original user query.
        documents (list): A list of Document objects (or (Document, score) tuples).
        model_name (str): The name of the Ollama model to use for answer generation.

    Returns:
        str: The consolidated answer.
    """
    if not documents:
        return "I could not find enough relevant information to answer your question."

    context_elements = []
    for item in documents:
        doc_content = None
        doc_metadata = {}

        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], 'page_content'):
            doc_content = item[0].page_content
            doc_metadata = item[0].metadata
        elif hasattr(item, 'page_content'):
            doc_content = item.page_content
            doc_metadata = item.metadata
        elif isinstance(item, str):
            doc_content = item

        if doc_content:
            # Check if the content is an image path
            if os.path.exists(doc_content) and doc_content.lower().endswith(('.png', '.jpg', '.jpeg')):
                context_elements.append(f"Image File Path: {doc_content}")
                print(f"DEBUG: Included image file path {os.path.basename(doc_content)} in context.")
            else:
                context_elements.append(doc_content)

    context = "\n\n---\n\n".join(context_elements)

    # Define the LLM for answer generation
    llm = ChatOllama(model=model_name)

    # Prompt for answer generation
    answer_prompt = ChatPromptTemplate.from_template(
        """You are a helpful veterinary assistant. Use the following retrieved information \
        to answer the user's question. If the information does not contain the answer, \
        please state that you cannot answer based on the provided context. If you see \
        a string that starts with "Image File Path:", you MUST include that entire \
        string exactly as it appears in your answer, along with any other relevant \
        information. Do not attempt to describe the image content if no image data is \
        present or if you cannot directly see the image.

        Question: {question}

        Context: {context}"""
    )

    # Create the answer generation chain
    answer_chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n🧠 Generating consolidated answer using model: {model_name}...")
    try:
        response = answer_chain.invoke({"question": query, "context": context})
        print("--- Consolidated Answer ---")
        print(response)
        return response
    except Exception as e:
        print(f"❌ An error occurred while generating the answer: {e}")
        return "An error occurred while generating the answer."

if __name__ == "__main__":
    chroma_db_folder = './textbook_test_example1/'
    query = "What would be the best way to pick up a cat? Can you show me images?"

    print("\n--- Querying 'summaries' collection and generating answer ---")
    summary_results = query_chromadb(query, chroma_db_folder, "summaries", k=3)
    generate_answer_from_docs(query, summary_results)

    print("\n\n--- Querying 'original_texts' collection and generating answer ---")
    original_text_results = query_chromadb(query, chroma_db_folder, "original_texts", k=3)
    generate_answer_from_docs(query, original_text_results)
