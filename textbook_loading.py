from typing import Any, Optional
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import re
import os
import uuid
import base64
import ollama
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document

class Element(BaseModel):
    type: str
    text: Any
    context: Optional[str] = None
    original_index: Optional[int] = None
class UnifiedRetriever:
        def __init__(self, vectorstore, docstore, id_key="doc_id"):
            self.vectorstore = vectorstore
            self.docstore = docstore
            self.id_key = id_key
            self._collection = docstore._collection

        def retrieve(self, query, k=5):
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            output = []
            for doc, score in results:
                doc_id = doc.metadata.get(self.id_key)
                try:
                    original = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
                    original_doc = original["documents"][0] if original["documents"] else None
                    original_meta = original["metadatas"][0] if original["metadatas"] else None
                except Exception as e:
                    original_doc = None
                    original_meta = None
                output.append({
                    "summary": doc.page_content,
                    "original": original_doc,
                    "original_metadata": original_meta,
                    "summary_metadata": doc.metadata,
                    "score": score
                })
            return output

def load_book(file_name, image_output_dir="./figures"):
    """
    Loads a PDF book and partitions its elements (text, tables, images) using Unstructured.
    
    Args:
        file_name (str): The path to the PDF file.
        image_output_dir (str, optional): Directory to save extracted images.
                                         Defaults to "./figures".
                                         
    Returns:
        list: A list of raw elements extracted from the PDF.
    """
    # Path to save images
    os.makedirs(image_output_dir, exist_ok=True)
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=file_name,
        languages=['eng'],
        strategy='hi_res',
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_output_dir=image_output_dir,
    )
    return raw_pdf_elements

def clean_and_categorize_elements(raw_pdf_elements, min_meaningful_text_length=20, window_size=1):
    """
    Cleans and categorizes raw PDF elements into texts, tables, and images.
    Also enriches image contexts by looking at surrounding text.

    Args:
        raw_pdf_elements (list): A list of raw elements obtained from partition_pdf.
        min_meaningful_text_length (int, optional): Minimum length for a text block to be considered meaningful. Defaults to 20.
        window_size (int, optional): Number of elements before and after an image to consider for context. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - texts (list): List of cleaned text chunks.
            - tables (list): List of extracted table contents.
            - images_raw (list): List of Element objects for images with enriched context.
            - headers_raw (list): List of raw header elements.
            - titles_raw (list): List of raw title elements.
            - footers_raw (list): List of raw footer elements.
            - figure_captions_raw (list): List of raw figure caption elements.
            - list_items_raw (list): List of raw list item elements.
    """
    text_for_semantic_chunking = []
    tables_raw = []
    images_raw = []
    headers_raw = []
    titles_raw = []
    footers_raw = []
    figure_captions_raw = []
    list_items_raw = []

    current_text_block = ""
    current_context_prefix = ""

    def finalize_text_block_inner():
        nonlocal current_text_block, current_context_prefix
        if current_text_block.strip() and len(current_text_block.strip()) >= min_meaningful_text_length:
            text_for_semantic_chunking.append(Element(type="text", text=current_text_block.strip()))
        current_text_block = ""

    for i, element in enumerate(raw_pdf_elements):
        element_type_str = str(type(element))
        element_text = str(element).strip()

        if "unstructured.documents.elements.Header" in element_type_str:
            finalize_text_block_inner()
            
            is_running_header = False
            lower_element_text = element_text.lower()
            if (
                "qxp" in lower_element_text or
                "pm" in lower_element_text or
                "am" in lower_element_text or
                "page" in lower_element_text or
                re.search(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', lower_element_text)
            ):
                is_running_header = True

            if not is_running_header:
                current_context_prefix = element_text + " "
            headers_raw.append(Element(type="header", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Title" in element_type_str:
            finalize_text_block_inner()
            current_context_prefix = element_text + " "
            titles_raw.append(Element(type="title", text=element_text, original_index=i))
        elif "unstructured.documents.elements.NarrativeText" in element_type_str or \
             "unstructured.documents.elements.ListItem" in element_type_str or \
             "unstructured.documents.elements.Text" in element_type_str:
            
            if len(element_text) < 5 and not any(char.isalpha() for char in element_text):
                continue

            if not current_text_block and current_context_prefix:
                current_text_block += current_context_prefix
                
            current_text_block += element_text + " "
            if "unstructured.documents.elements.ListItem" in element_type_str:
                list_items_raw.append(Element(type="list_item", text=element_text, original_index=i))

        elif "unstructured.documents.elements.Table" in element_type_str:
            finalize_text_block_inner()
            tables_raw.append(Element(type="table", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Image" in element_type_str:
            finalize_text_block_inner()

            image_path = getattr(element.metadata, "image_path", "N/A")
            images_raw.append(Element(type="image", text=image_path, context="", original_index=i))
            current_text_block = ""

        elif "unstructured.documents.elements.FigureCaption" in element_type_str:
            finalize_text_block_inner()
            figure_captions_raw.append(Element(type="figure_caption", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Footer" in element_type_str:
            footers_raw.append(Element(type="footer", text=element_text, original_index=i))

    finalize_text_block_inner()

    enrich_image_context(images_raw, raw_pdf_elements, window_size=window_size)

    texts = [e.text for e in text_for_semantic_chunking]
    tables = [e.text for e in tables_raw]
    
    return texts, tables, images_raw, headers_raw, titles_raw, footers_raw, figure_captions_raw, list_items_raw

def enrich_image_context(images, all_raw_elements, window_size=1):
    """
    Enriches the context for each image by looking at a window of surrounding text and captions.

    Args:
        images (list): A list of image Element objects to enrich.
        all_raw_elements (list): All raw elements from the PDF, used to find surrounding text.
        window_size (int, optional): Number of elements to look before and after the image. Defaults to 3.
    """
    for img_element in images:
        img_index = img_element.original_index
        if img_index is None:
            continue

        start_index = max(0, img_index - window_size)
        end_index = min(len(all_raw_elements), img_index + window_size + 1)
        
        surrounding_text_elements = []
        for j in range(start_index, end_index):
            surrounding_element = all_raw_elements[j]
            element_type_str = str(type(surrounding_element))
            element_text = str(surrounding_element).strip()

            if "unstructured.documents.elements.NarrativeText" in element_type_str or \
               "unstructured.documents.elements.ListItem" in element_type_str or \
               "unstructured.documents.elements.Text" in element_type_str or \
               "unstructured.documents.elements.FigureCaption" in element_type_str:
                
                if len(element_text) >= 5 or any(char.isalpha() for char in element_text):
                     surrounding_text_elements.append(element_text)
            
            if "unstructured.documents.elements.Header" in element_type_str or \
               "unstructured.documents.elements.Title" in element_type_str:
                lower_element_text = element_text.lower()
                is_running_header = (
                    "qxp" in lower_element_text or "pm" in lower_element_text or
                    "am" in lower_element_text or "page" in lower_element_text or
                    re.search(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', lower_element_text)
                )
                if not is_running_header:
                    surrounding_text_elements.append(element_text)


        enriched_context = " ".join(surrounding_text_elements).strip()
        if not enriched_context:
            enriched_context = "No specific text context available around this image."
        
        img_element.context = enriched_context

def enrich_table_context(tables, all_raw_elements, window_size=1):
    """
    Enriches the context for each table by looking at a window of surrounding text and captions.

    Args:
        tables (list): A list of table Element objects to enrich.
        all_raw_elements (list): All raw elements from the PDF, used to find surrounding text.
        window_size (int, optional): Number of elements to look before and after the table. Defaults to 1.
    """
    for tbl_element in tables:
        tbl_index = tbl_element.original_index
        if tbl_index is None:
            continue
        start_index = max(0, tbl_index - window_size)
        end_index = min(len(all_raw_elements), tbl_index + window_size + 1)
        surrounding_text_elements = []
        for j in range(start_index, end_index):
            surrounding_element = all_raw_elements[j]
            element_type_str = str(type(surrounding_element))
            element_text = str(surrounding_element).strip()
            if "unstructured.documents.elements.NarrativeText" in element_type_str or \
               "unstructured.documents.elements.ListItem" in element_type_str or \
               "unstructured.documents.elements.Text" in element_type_str or \
               "unstructured.documents.elements.FigureCaption" in element_type_str:
                if len(element_text) >= 5 or any(char.isalpha() for char in element_text):
                    surrounding_text_elements.append(element_text)
            if "unstructured.documents.elements.Header" in element_type_str or \
               "unstructured.documents.elements.Title" in element_type_str:
                lower_element_text = element_text.lower()
                is_running_header = (
                    "qxp" in lower_element_text or "pm" in lower_element_text or
                    "am" in lower_element_text or "page" in lower_element_text or
                    re.search(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', lower_element_text)
                )
                if not is_running_header:
                    surrounding_text_elements.append(element_text)
        enriched_context = " ".join(surrounding_text_elements).strip()
        if not enriched_context:
            enriched_context = "No specific text context available around this table."
        tbl_element.context = enriched_context

def summarize_elements(texts, tables, images_raw, raw_pdf_elements=None):
    """
    Summarizes text, table, and relevant image elements using Ollama models.
    Also performs an image relevance check to filter out irrelevant images before summarization.

    Args:
        texts (list): List of text chunks to summarize.
        tables (list): List of table Element objects to summarize (with context).
        images_raw (list): List of Element objects for images with enriched context.
        raw_pdf_elements (list, optional): All raw elements for context enrichment.

    Returns:
        tuple: A tuple containing:
            - text_summaries (list): Summaries of text chunks.
            - table_summaries (list): Summaries of table contents.
            - img_summaries (list): Summaries of relevant images.
            - image_paths (list): Paths of the relevant images.
            - relevant_images_to_summarize (list): Element objects of images deemed relevant.
    """
    model = ChatOllama(model="llama3.2")

    prompt_text_summary = "You are an assistant tasked with concisely summarizing text sections related to veterinary advice and pet care. Focus on key information, main ideas, and any actionable advice. Just give me the summary, be concise and do not be verbose. Text chunk: {element} "
    prompt_text = ChatPromptTemplate.from_template(prompt_text_summary)
    text_summarize_chain = {"element": lambda x: x} | prompt_text | model | StrOutputParser()

    # Enrich table context if not already done
    if raw_pdf_elements is not None and tables and hasattr(tables[0], 'context'):
        enrich_table_context(tables, raw_pdf_elements, window_size=1)

    prompt_table_summary = (
        "You are an assistant tasked with extracting key information, trends, and important numerical data from the provided table, "
        "especially as it relates to veterinary topics, animal health, or clinical practice. Use the provided context to help interpret the table. "
        "Just give me the summary, be concise and do not be verbose.\n\n"
        "Context: {context}\n"
        "Table chunk: {element}"
    )
    prompt_table = ChatPromptTemplate.from_template(prompt_table_summary)
    table_summarize_chain = (
        {"element": lambda x: x["element"], "context": lambda x: x["context"]}
        | prompt_table
        | model
        | StrOutputParser()
    )
    # Prepare table-context pairs
    table_context_pairs = []
    for tbl in tables:
        context = getattr(tbl, 'context', "")
        table_context_pairs.append({"element": tbl.text if hasattr(tbl, 'text') else tbl, "context": context})

    text_summaries = text_summarize_chain.batch(texts, {"max_concurrency": 8})
    table_summaries = table_summarize_chain.batch(table_context_pairs, {"max_concurrency": 8})

    relevant_images_to_summarize = []
    print("Checking image relevance with local textual context...")
    for image_element in images_raw:
        image_filename = image_element.text
        image_context = image_element.context

        if not image_context:
            image_context = "No specific text context was captured for this image, infer relevance from filename."

        messages_for_ollama = [
        {
            "role": "user",
            "content": (
                "You are a veterinary assistant helping to filter images for a veterinary knowledge base. "
                "Given the following local textual context and the image filename, decide if the image is relevant to veterinary topics, animal care, or pet health. "
                "If the context is clearly nonsense, random symbols, or not related to veterinary medicine (e.g., '--o--', '*92312*))____++', or similar gibberish), respond with 'no'. "
                "If the image and context are relevant to veterinary topics, respond with 'yes'. "
                "Only respond with 'yes' or 'no'.\n\n"
                f"Local Textual Context: {image_context}\n"
                f"Image Filename: {os.path.basename(image_filename)}"
            ),
            "images": []
            }
        ]

        if os.path.exists(image_filename):
            messages_for_ollama[0]["images"].append(image_filename)
        else:
            print(f"WARNING: Image file not found: {image_filename}. Cannot pass image data to model.")

        response_content = "no"
        try:
            response_obj = ollama.chat(
                model="minicpm-v:8b", # Or your preferred vision model
                messages=messages_for_ollama,
                options={"temperature": 0.0}
            )
            response_content = response_obj['message']['content']
        except Exception as e:
            print(f"ERROR: Failed to invoke Ollama for image relevance check on {image_filename}: {e}")
            response_content = "no"

        if "yes" in response_content.lower().strip():
            relevant_images_to_summarize.append(image_element)
        else:
            print(f"Skipping irrelevant image: {image_filename}")

    print(f"Number of relevant images for summarization: {len(relevant_images_to_summarize)}")

    # Generating summaries for those relevant images.
    img_summaries = []
    image_paths = []

    for img_element in relevant_images_to_summarize:
        image_path = img_element.text
        context = img_element.context or ""
        if os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                prompt = (
                    "From a veterinary point of view, explain what this image is about in the provided context. "
                    "Focus on the specific actions, techniques, or procedures depicted, and relate them to the veterinary scenario described in the context. "
                    "Be concise, use direct language, and include likely search terms based on the context.\n\n"
                    "Avoid pleasantries"
                    f"Context: {context}"
                )
                response = ollama.chat(
                    model='minicpm-v:8b',
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [image_data]
                        }
                    ]
                )
                summary = response['message']['content']
                img_summaries.append(summary)
                image_paths.append(image_path)
            except Exception as e:
                print(f"Error summarizing image {image_path}: {e}")
        else:
            print(f"Image file not found for summarization: {image_path}")

    return text_summaries, table_summaries, img_summaries, image_paths, relevant_images_to_summarize

def store_in_chromadb(text_summaries, texts, table_summaries, tables, img_summaries, image_paths, persist_directory="./chroma_db"):
    """
    Stores the summarized text, table, and image data into a ChromaDB vector store on disk.

    Args:
        text_summaries (list): Summaries of text chunks.
        texts (list): Original text chunks (parent documents).
        table_summaries (list): Summaries of table contents.
        tables (list): Original table contents (parent documents).
        img_summaries (list): Summaries of relevant images.
        image_paths (list): Paths of the relevant images (parent documents).
        persist_directory (str, optional): The directory where the ChromaDB collection will be persisted.
                                          Defaults to "./chroma_db".

    Returns:
        UnifiedRetriever: An initialized retriever object for all modalities.
    """
    open_clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

    # Vectorstore for summaries (for similarity search)
    summary_vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=open_clip_embeddings,
        persist_directory=persist_directory
    )
    # Persistent docstore for originals (all modalities)
    docstore = Chroma(
        collection_name="originals",
        embedding_function=open_clip_embeddings,
        persist_directory=persist_directory
    )
    id_key = "doc_id"

    # Store text chunks
    if texts:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i], "type": "text"})
            for i, s in enumerate(text_summaries)
        ]
        original_text_docs = [
            Document(page_content=texts[i], metadata={id_key: doc_ids[i], "type": "text"})
            for i in range(len(texts))
        ]
        summary_vectorstore.add_documents(summary_texts, ids=doc_ids)
        docstore.add_documents(original_text_docs, ids=doc_ids)

    # Store tables as JSON
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i], "type": "table"})
            for i, s in enumerate(table_summaries)
        ]
        original_table_docs = [
            Document(page_content=json.dumps(tables[i]), metadata={id_key: table_ids[i], "type": "table"})
            for i in range(len(tables))
        ]
        summary_vectorstore.add_documents(summary_tables, ids=table_ids)
        docstore.add_documents(original_table_docs, ids=table_ids)

    # Store image summaries and originals (as path)
    if img_summaries:
        img_ids = [str(uuid.uuid4()) for _ in img_summaries]
        summary_img_docs = [
            Document(page_content=img_summaries[i], metadata={id_key: img_ids[i], "type": "image", "image_path": image_paths[i]})
            for i in range(len(img_summaries))
        ]
        original_img_docs = [
            Document(page_content=image_paths[i], metadata={id_key: img_ids[i], "type": "image"})
            for i in range(len(image_paths))
        ]
        summary_vectorstore.add_documents(summary_img_docs, ids=img_ids)
        docstore.add_documents(original_img_docs, ids=img_ids)

    return UnifiedRetriever(summary_vectorstore, docstore, id_key)

def delete_irrelevant_images(images_raw, relevant_images_to_summarize):
    """
    Deletes image files that were identified as irrelevant during the summarization process.

    Args:
        images_raw (list): All original image Element objects.
        relevant_images_to_summarize (list): Element objects of images deemed relevant and summarized.
    """
    relevant_image_paths = {img_elem.text for img_elem in relevant_images_to_summarize}
    images_deleted_count = 0
    for image_element in images_raw:
        image_path = image_element.text
        if image_path not in relevant_image_paths:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Successfully deleted irrelevant image: {image_path}")
                    images_deleted_count += 1
                except OSError as e:
                    print(f"Error deleting file {image_path}: {e}")
            else:
                print(f"Skipping deletion: Image file not found at {image_path}")
    print(f"Finished deleting images. Total deleted: {images_deleted_count}")






