# LLM Veterinary AI: Multimodal Vet Assistant

## Project Overview

**LLM Veterinary AI** is an open-source project building an intelligent veterinary assistant powered by Retrieval-Augmented Generation (RAG) with multimodal capabilities. This system answers veterinary questions using information from text, images, tables, and more—delivering accurate, context-rich responses for veterinary use cases.

## Features

- **Multimodal RAG Pipeline**
  - Extracts and summarizes content from PDFs, including text, tables, and images (using `unstructured`).
  - Generates image summaries with vision-capable LLMs (e.g., Llama3.2-vision, Llava).
  - Stores all content (text summaries, table summaries, image summaries) in a vector database for semantic retrieval.
- **Flexible Retrieval**
  - Uses semantic search to find and combine the most relevant text, tables, and images for a given query.
  - Dynamically presents the answer, including images, in Markdown format.
- **Veterinary Domain Focus**
  - Example: "How do I pick up a cat?" returns both procedural text and illustrative images drawn from the dataset.
- **Extensible Design**
  - Easily add new data types and integrate more advanced models in the future.

## How It Works

1. **Data Ingestion**
    - PDFs are partitioned into text, tables, and images.
    - Each segment is summarized using LLMs for concise storage and retrieval.

2. **Image Processing**
    - Images are described by vision models; summaries are saved alongside the originals.

3. **Vector Store**
    - All summaries (text, tables, images) are embedded and stored in a vector DB (Chroma + in-memory store).

4. **Retrieval & Answer Generation**
    - The retriever locates the most relevant text, tables, and images for a user query.
    - The multimodal LLM combines these elements to generate a comprehensive, illustrated answer.
    - Responses include Markdown-rendered images where helpful.

## Example

**User Query:**  
`Can you show me how to pick up a cat? Include images.`

**Sample Output:**  
- Step-by-step instructions (with safety tips for cooperative, apprehensive, or frightened cats).
- Embedded images illustrating proper techniques.

## Dependencies

- Python 3.x
- Jupyter Notebook
- Key libraries:
  - `langchain`, `langchain-chroma`
  - `unstructured[all-docs]`
  - `pydantic`, `lxml`
  - Vision model support: `ollama` (for Llama3.2-vision, Llava, etc.)
  - `Chroma`, `GPT4AllEmbeddings`, `Pillow`

Install the main dependencies with:

```bash
pip install langchain langchain-chroma "unstructured[all-docs]" pydantic lxml
