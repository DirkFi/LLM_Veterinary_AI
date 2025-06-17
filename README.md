# 🐾 Veterinary Textbook RAG Pipeline

This project is a Retrieval-Augmented Generation (RAG) pipeline for veterinary textbooks. It enables semantic search and retrieval over both text and images, allowing users to query for veterinary advice, procedures, and visual references using natural language.

## Features

- **PDF Ingestion:** Partition and extract text, tables, and images from veterinary textbooks.
- **Contextual Chunking:** Text and images are chunked and enriched with surrounding context for better retrieval.
- **Image Summarization:** Images are summarized using a vision-language model, with summaries tied to their local text context.
- **Multimodal Embeddings:** Both text and image summaries are embedded into a shared vector space using OpenCLIP, enabling cross-modal retrieval.
- **Semantic Search:** Users can search using natural language and retrieve relevant text, tables, and images.
- **Original Content Retrieval:** Each summary is linked to its original full text, table, or image for detailed reference.
- **Irrelevant Image Filtering:** Only images relevant to veterinary content are kept and summarized.

## How It Works

1. **Ingestion:**  
   - Load a PDF and extract its elements (text, tables, images).
   - Clean and chunk the text, enrich image context, and filter irrelevant images.
   - Summarize text, tables, and images (with context-aware prompts for images).
   - Store summaries and originals in ChromaDB vector stores using OpenCLIP embeddings.

2. **Retrieval:**  
   - User submits a natural language query.
   - The query is embedded with OpenCLIP and used to search the summary vector store.
   - Top results (text, table, or image summaries) are returned, with links to the original content.

## Example Query

> "How do I safely pick up a cat for examination?"

- Returns relevant text instructions, tables, and images (with summaries) about cat handling and restraint.

## Setup

1. **Install dependencies:**
   ```bash
   pip install langchain langchain-experimental langchain-chroma pillow open_clip_torch torch matplotlib unstructured pydantic
   ```

2. **Prepare your data:**
   - Place your PDF(s) in the `./data/` directory.

3. **Run the ingestion pipeline:**
   ```python
   # Example usage in a script or notebook
   from textbook_loading import run_ingestion
   retriever = run_ingestion('./data/your_textbook.pdf')
   ```

4. **Query the retriever:**
   ```python
   query = "How to pick up a cooperative cat?"
   docs = retriever.invoke(query, k=8)
   for doc in docs:
       print(doc.page_content)
       # Use doc.metadata['image_path'] to display images if needed
   ```

## Customization

- **Image Summarization Prompt:**  
  The prompt for image summarization is context-aware and can be customized in `textbook_loading.py` for your specific needs.

- **Chunking Window Size:**  
  Adjust the `window_size` parameter in `clean_and_categorize_elements` for more or less context around images.

## Notes

- **OpenCLIP** is used for all embeddings, ensuring text and images are in the same vector space.
- **ChromaDB** is used for fast vector search and persistent storage.
- **Ollama** and vision-language models (e.g., `llava:7b`, `minicpm-v:8b`) are used for image relevance and summarization.

## Troubleshooting

- If you change your chunking, summarization, or ingestion logic, **delete the ChromaDB directory and re-ingest** to avoid stale or mismatched data.
- Make sure all dependencies are installed and your models are available locally.

## License

MIT License
