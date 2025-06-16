import os
from textbook_loading import load_book, clean_and_categorize_elements, summarize_elements, store_in_chromadb, delete_irrelevant_images

def run_ingestion(file_path, image_output_dir="./figures", chroma_persist_dir="./chroma_db"):
    """
    Orchestrates the entire data ingestion pipeline for a PDF textbook.
    This includes loading, cleaning, categorizing, summarizing, and storing data into ChromaDB,
    and finally deleting irrelevant images.

    Args:
        file_path (str): The path to the PDF file to be ingested.
        image_output_dir (str, optional): Directory to save extracted images. Defaults to "./figures/shortExample1/".
        chroma_persist_dir (str, optional): The directory where the ChromaDB collection will be persisted. Defaults to "./chroma_db".

    Returns:
        MultiVectorRetriever: An initialized MultiVectorRetriever object containing the ingested data.
    """
    print(f"✨ --- Starting ingestion for {file_path} --- ✨")

    # 1. Load the book and partition elements
    print("📚 Loading and partitioning PDF elements...")
    raw_pdf_elements = load_book(file_path, image_output_dir)
    print(f"✅ Found {len(raw_pdf_elements)} raw elements.")

    # 2. Clean and categorize elements
    print("🧹 Cleaning and categorizing elements...")
    texts, tables, images_raw, _, _, _, _, _ = clean_and_categorize_elements(raw_pdf_elements)
    print(f" categorize: {len(texts)} text chunks, {len(tables)} tables, {len(images_raw)} images.")

    # 3. Summarize elements and perform image relevance check
    print("📝 Summarizing text, tables, and relevant images...")
    text_summaries, table_summaries, img_summaries, image_paths, relevant_images_to_summarize = summarize_elements(texts, tables, images_raw)
    print(f"📊 Summarized: {len(text_summaries)} text summaries, {len(table_summaries)} table summaries, {len(img_summaries)} image summaries.")

    # 4. Store in ChromaDB
    print(f"💾 Storing data into ChromaDB at {chroma_persist_dir}...")
    retriever = store_in_chromadb(text_summaries, texts, table_summaries, tables, img_summaries, image_paths, persist_directory=chroma_persist_dir)
    print("🎉 Data ingestion complete. Retriever initialized.")

    # --- Temporary Inspection Code (for verification) ---
    print("\n🔍 Verifying ChromaDB contents via retriever:")
    # Check total documents in the vectorstore (summaries)
    vectorstore_ids = retriever.vectorstore.get(include=[])['ids']
    print(f"Total documents in vectorstore (summaries): {len(vectorstore_ids)}")

    # Check total documents in the docstore (original content)
    # Note: InMemoryStore does not have a direct 'get_all_ids' or 'count' method like Chroma
    # We'll assume if vectorstore has IDs, the docstore was populated correspondingly.
    # For a direct check, you'd iterate through docstore.mget(vectorstore_ids)
    docstore_keys = retriever.docstore.mget(vectorstore_ids)
    print(f"Total documents in docstore (original content): {len(docstore_keys)}")

    if len(vectorstore_ids) > 0:
        print("Sample of docstore original content (first 1 document if available):")
        # Retrieve the actual document content for a few IDs
        sample_doc_ids = vectorstore_ids[:1]
        retrieved_docs = retriever.docstore.mget(sample_doc_ids)
        for i, doc_content in enumerate(retrieved_docs):
            if doc_content:
                print(f"--- Document {i+1} Content (from docstore) ---")
                print(str(doc_content)[:500] + "...") # Print first 500 chars
            else:
                print(f"Could not retrieve content for doc ID: {sample_doc_ids[i]}")
    else:
        print("No documents to sample from.")
    print("--- End of Inspection ---\n")
    # ---------------------------------------------------

    # 5. Delete irrelevant images
    print("🗑️ Deleting irrelevant images...")
    delete_irrelevant_images(images_raw, relevant_images_to_summarize)

    return retriever

if __name__ == "__main__":
    # Example usage:
    #pdf_file = './data/shortExample2_Nutrition_20Pgs.pdf'
    pdf_file = './data/shortExample1P_PickupCat.pdf'
    output_images_folder = './figures'
    chroma_db_folder = './textbook_test_example1/' 

    # Make sure the data directory exists
    if not os.path.exists('./data'):
        print("Error: './data' directory not found. Please ensure your PDF is in the correct location.")
    elif not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}. Please check the path.")
    else:
        retriever = run_ingestion(pdf_file, output_images_folder, chroma_db_folder)
        print("\nIngestion process finished. You can now use the 'retriever' object for queries.")
        # Example query (this would typically be in another script or part of your application)
        query = "How to pick up a cooperative cat?"
        docs = retriever.invoke(query)
        print(f"\nRetrieved documents for query '{query}':")
        for doc in docs:
            if hasattr(doc, 'page_content'):
                print(doc.page_content[:200] + "...")
            else:
                print(str(doc)[:200] + "...") 