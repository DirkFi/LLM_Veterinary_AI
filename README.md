# 🐾 Veterinary Textbook Multimodal RAG Pipeline

This project implements a modern Retrieval-Augmented Generation (RAG) pipeline for veterinary textbooks, supporting both text and images. It enables semantic search and retrieval over multimodal content, allowing users to query for veterinary advice, procedures, and visual references using natural language.

## Pipeline Overview

### 1. Ingestion & Vector Database Construction
- **Raw Extraction:**
  - PDFs are parsed to extract all elements: text blocks, tables, and images (with their local context).
- **Cleaning & Chunking:**
  - Text is cleaned, junk/irrelevant content is filtered, and semantically chunked for optimal retrieval.
  - Images are assigned context using a hybrid of captions, local paragraphs, and semantic similarity.
- **Image Summarization:**
  - Each image is summarized using a vision-language model (VLM), with summaries enriched by local context.
- **Multimodal Embedding:**
  - Both text chunks and image summaries are embedded into a shared vector space using OpenCLIP.
- **Storage:**
  - All summaries (text, table, image) and their metadata are stored in a ChromaDB vector database.
  - Each summary links to its original content (full text, table, or image path) for reference.

### 2. Query Handling with LangGraph
- **User Query Intake:**
  - User submits a natural language query (optionally with an image).
- **Graph-based Processing:**
  - The query is routed through a LangGraph workflow:
    - **Domain Classification:** Determines if the query is veterinary, emergency, or irrelevant.
    - **Query Refinement:** Expands and clarifies the query, optionally using image context.
    - **Decomposition:** Breaks complex queries into focused sub-queries (including visual sub-queries).
    - **Contextual Retrieval:** Each sub-query retrieves top-k relevant text and image summaries from the vector DB.
    - **Multimodal Reranking:** Retrieved results are reranked using a cross-encoder or vision-language reranker (e.g., JinaAI/jina-reranker-m0), supporting both text and images for improved relevance.
    - **Answer Synthesis:** The top reranked results are synthesized into a final answer, optionally with follow-up suggestions.

## Example Workflow

1. **Ingestion:**
   - Load a PDF, extract and clean all elements, chunk text, summarize images, embed, and store in ChromaDB.
2. **User Query:**
   - "What are the signs of otitis in cats?"
3. **LangGraph Flow:**
   - Classify → Refine → Decompose → Retrieve (text + images) → Rerank → Synthesize answer.
4. **Output:**
   - Returns the most relevant text and image results, with links to original textbook content.

## Key Features
- **Multimodal:** Unified retrieval and reranking for both text and images.
- **Context-aware:** Image summaries and retrieval are enriched by local and semantic context.
- **Flexible Reranking:** Supports advanced rerankers (e.g., JinaAI/jina-reranker-m0) for improved result quality.
- **Traceable:** Every summary links back to its original textbook element.

## Notes
- **Re-ingest** if you change chunking, summarization, or ingestion logic to avoid stale data in ChromaDB.
- **Dependencies:** Requires OpenCLIP, ChromaDB, LangChain, and a vision-language model for image summarization and reranking.

## License

MIT License
