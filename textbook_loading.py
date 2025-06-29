from typing import Any, Optional
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import re
import os
import uuid
import base64
import ollama
import json
from PIL import Image

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.documents import Document
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class Element(BaseModel):
    type: str
    text: Any
    context: Optional[str] = None
    original_index: Optional[int] = None

class UnifiedRetriever:
    """
    UnifiedRetriever supports multi-modal retrieval from the vectorstore and docstore.
    It can retrieve by text, image, or both (multi-query), and supports metadata filtering by modality.
    """
    def __init__(self, vectorstore, docstore, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.id_key = id_key
        self._collection = docstore._collection

    def retrieve(self, query, k=5, filter=None):
        """
        Retrieve top-k results for a query, optionally filtered by metadata (e.g., modality).
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
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

    def retrieve_multi_modal(self, query, k=5, text_types=("text",), image_types=("image", "image_summary")):
        """
        Multi-Query/Multi-Modal Retrieval:
        - Retrieves top-k text and top-k image/image_summary results for the query.
        - Merges and sorts by score.
        - Returns a list of results with modality info.
        """
        # Retrieve text results
        text_results = self.vectorstore.similarity_search_with_score(query, k=k, filter={"type": {"$in": list(text_types)}})
        # Retrieve image/image_summary results
        image_results = self.vectorstore.similarity_search_with_score(query, k=k, filter={"type": {"$in": list(image_types)}})
        # Merge and sort by score (lower is better if using distance, higher is better if using similarity)
        all_results = []
        for doc, score in text_results:
            doc_id = doc.metadata.get(self.id_key)
            all_results.append({
                "modality": doc.metadata.get("type"),
                "summary": doc.page_content,
                "original_metadata": doc.metadata,
                "score": score,
                "doc_id": doc_id
            })
        for doc, score in image_results:
            doc_id = doc.metadata.get(self.id_key)
            all_results.append({
                "modality": doc.metadata.get("type"),
                "summary": doc.page_content,
                "original_metadata": doc.metadata,
                "score": score,
                "doc_id": doc_id
            })
        # Sort by score (descending if similarity, ascending if distance)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

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

def is_junk_text(text):
    t = text.strip()
    if not t or len(t) < 5:
        return True
    junk_patterns = [
        r'^—o—$', r'^\*+$', r'^_+$', r'^page \d+', r'^\d{1,2}/\d{1,2}/\d{2,4}$',
        r"^cat owner[’']?s home veterinary handbook$", r'^\d+$', r'^ch\d+', r'^fig(ure)?[-\d]+',
        r'^04_095300.*page.*$', r'^[\W_]+$'
    ]
    for pat in junk_patterns:
        if re.match(pat, t, re.IGNORECASE):
            return True
    # If mostly non-alphabetic
    if len(t) > 0 and sum(c.isalpha() for c in t) < 0.3 * len(t):
        return True
    return False

def enrich_image_context(images_raw, raw_pdf_elements, window_size=2):
    """
    Assigns high-quality, boundary-aware context to each image in images_raw.
    - Looks at a window of elements before and after the image.
    - Stops at Title or Header (section/chapter boundary).
    - Prioritizes FigureCaption if present.
    - Skips headers, footers, and junk text.
    - Prefers preceding context if the image is at a boundary.
    """
    for img_elem in images_raw:
        idx = img_elem.original_index
        if idx is None:
            continue
        context_texts = []
        found_caption = None
        # Look backwards for context
        for j in range(idx-1, max(-1, idx-window_size-1), -1):
            el = raw_pdf_elements[j]
            el_type = str(type(el))
            el_text = str(el).strip()
            if "Title" in el_type or "Header" in el_type:
                break  # Stop at section/chapter boundary
            if "FigureCaption" in el_type and not is_junk_text(el_text):
                found_caption = el_text
                break  # Use caption as main context
            if ("NarrativeText" in el_type or "ListItem" in el_type or "Text" in el_type) and not is_junk_text(el_text):
                context_texts.insert(0, el_text)  # Prepend
        # Look forwards for context, but stop at Title/Header
        for j in range(idx+1, min(len(raw_pdf_elements), idx+window_size+1)):
            el = raw_pdf_elements[j]
            el_type = str(type(el))
            el_text = str(el).strip()
            if "Title" in el_type or "Header" in el_type:
                break
            if "FigureCaption" in el_type and not is_junk_text(el_text):
                found_caption = el_text
                break
            if ("NarrativeText" in el_type or "ListItem" in el_type or "Text" in el_type) and not is_junk_text(el_text):
                context_texts.append(el_text)
        # Assign context
        if found_caption:
            img_elem.context = found_caption
        elif context_texts:
            img_elem.context = " ".join(context_texts)
        else:
            img_elem.context = "No specific text context available around this image."

def clean_and_categorize_elements(raw_pdf_elements, min_meaningful_text_length=15, window_size=1):
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

    def finalize_text_block_inner(last_index=None):
        nonlocal current_text_block, current_context_prefix
        cleaned = current_text_block.strip()
        if cleaned and len(cleaned) >= min_meaningful_text_length and not is_junk_text(cleaned):
            text_for_semantic_chunking.append(Element(type="text", text=cleaned, original_index=last_index))
        current_text_block = ""

    for i, element in enumerate(raw_pdf_elements):
        element_type_str = str(type(element))
        element_text = str(element).strip()

        if "unstructured.documents.elements.Header" in element_type_str:
            finalize_text_block_inner(i)
            
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

            if not is_running_header and not is_junk_text(element_text):
                current_context_prefix = element_text + " "
            headers_raw.append(Element(type="header", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Title" in element_type_str:
            finalize_text_block_inner(i)
            if not is_junk_text(element_text):
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
            if "unstructured.documents.elements.ListItem" in element_type_str and not is_junk_text(element_text):
                list_items_raw.append(Element(type="list_item", text=element_text, original_index=i))

        elif "unstructured.documents.elements.Table" in element_type_str:
            finalize_text_block_inner(i)
            if not is_junk_text(element_text):
                tables_raw.append(Element(type="table", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Image" in element_type_str:
            finalize_text_block_inner(i)

            image_path = getattr(element.metadata, "image_path", "N/A")
            images_raw.append(Element(type="image", text=image_path, context="", original_index=i))
            current_text_block = ""

        elif "unstructured.documents.elements.FigureCaption" in element_type_str:
            finalize_text_block_inner(i)
            if not is_junk_text(element_text):
                figure_captions_raw.append(Element(type="figure_caption", text=element_text, original_index=i))
        elif "unstructured.documents.elements.Footer" in element_type_str:
            if not is_junk_text(element_text):
                footers_raw.append(Element(type="footer", text=element_text, original_index=i))

    finalize_text_block_inner(i)
    # Enrich image context with boundary-aware logic
    enrich_image_context(images_raw, raw_pdf_elements, window_size=window_size)
    texts = text_for_semantic_chunking
    tables = tables_raw
    
    return texts, tables, images_raw, headers_raw, titles_raw, footers_raw, figure_captions_raw, list_items_raw

def semantic_chunk_texts(texts, embedding_model, n_clusters=30):
    """
    Semantically chunk the cleaned text into coherent groups using embeddings and clustering.
    Returns a list of merged, semantically coherent text chunks.
    """
    if len(texts) == 0:
        return []
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings)
    n_clusters = min(n_clusters, len(texts))
    if n_clusters < 2:
        return texts
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(embeddings)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(texts[idx])
    semantic_chunks = ['\n'.join(cluster) for cluster in clusters.values()]
    return semantic_chunks

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

def is_decorative_image(image_path, min_area=500, extreme_ratio=8):
    """
    Returns True if the image is likely a divider/decorative element.
    - min_area: images smaller than this (in pixels) are likely not informative.
    - extreme_ratio: if width/height or height/width exceeds this, likely a divider.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            area = w * h
            if area < min_area:
                return True
            ratio = max(w/h, h/w)
            if ratio > extreme_ratio:
                return True
    except Exception:
        pass
    return False

def summarize_elements(texts, tables, images_raw, raw_pdf_elements=None):
    """
    Summarizes text, table, and relevant image elements using Ollama models.
    Also performs an image relevance check to filter out irrelevant images before summarization.
    For images, both the image and its LLM-generated summary are embedded separately for multi-modal retrieval.
    
    Returns:
        tuple: (text_summaries, table_summaries, image_paths, relevant_images, image_summaries)
    """
    model = ChatOllama(model="llama3.2:3b")
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
    table_context_pairs = []
    for tbl in tables:
        context = getattr(tbl, 'context', "")
        table_context_pairs.append({"element": tbl.text if hasattr(tbl, 'text') else tbl, "context": context})
    text_summaries = text_summarize_chain.batch(texts, {"max_concurrency": 8})
    table_summaries = table_summarize_chain.batch(table_context_pairs, {"max_concurrency": 8})
    print("Texts and Tables Summary Done!")
    
    # Image Handling, filtering out irrelevant imgs
    relevant_images = []
    image_summaries = []
    print("Checking image relevance with local textual context...")
    for image_element in images_raw:
        image_filename = image_element.text
        image_context = image_element.context
        # Decorative image filtering
        if is_decorative_image(image_filename):
            print(f"Skipping decorative image: {image_filename}")
            continue
        if not image_context:
            image_context = "No specific text context was captured for this image, infer relevance from filename."
        messages_for_ollama = [
        {
            "role": "user",
            "content": (
                "You are a veterinary assistant helping to filter images for a veterinary knowledge base.\n"
                "Given the following local textual context and the image, answer with 'yes' if the image is a real photograph, diagram, or illustration relevant to veterinary medicine, animal care, or pet health.\n"
                "If the image is a decorative divider, border, simple geometric shape, or contains no meaningful content, answer with 'no'.\n"
                "Only respond with 'yes' or 'no'.\n\n"
                f"Local Textual Context: {image_context}\n"
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
            relevant_images.append(image_element)
        else:
            print(f"Skipping irrelevant image: {image_filename}")
    print(f"Number of relevant images: {len(relevant_images)}")
    image_paths = [img_elem.text for img_elem in relevant_images]
    # Generate LLM summaries for each relevant image
    print("Generating LLM summaries for relevant images...")
    for img_elem in relevant_images:
        image_filename = img_elem.text
        image_context = img_elem.context or "No specific text context was captured for this image."
        # Compose a prompt for the LLM to summarize the image using both the image and its context
        img_summary_prompt = (
            "You are a veterinary assistant AI. Given the following image and its local text context, "
            "write a concise, factual summary of what is depicted in the image, focusing on veterinary relevance.\n"
            "If the image is a decorative divider, border, or contains no meaningful content, respond with: 'This image is a decorative or non-informative element and does not contain veterinary-relevant content.'\n"
            "If the image shows a procedure, anatomy, or a specific condition, describe it clearly.\n"
            "If the context provides extra clues, use them.\n\n"
            f"Local Text Context: {image_context}\n"
            f"Image Filename: {os.path.basename(image_filename)}\n"
            "Image summary:"
        )
        img_messages = [{
            "role": "user",
            "content": img_summary_prompt,
            "images": [image_filename] if os.path.exists(image_filename) else []
        }]
        try:
            img_summary_response = ollama.chat(
                model="minicpm-v:8b",
                messages=img_messages,
                options={"temperature": 0.2}
            )
            img_summary = img_summary_response['message']['content'].strip()
        except Exception as e:
            print(f"ERROR: Failed to summarize image {image_filename}: {e}")
            img_summary = image_context  # fallback to context
        image_summaries.append(img_summary)
    return text_summaries, table_summaries, image_paths, relevant_images, image_summaries

def store_in_chromadb(text_summaries, texts, table_summaries, tables, image_paths, relevant_images=None, image_summaries=None, persist_directory="./chroma_db"):
    """
    Stores the summarized text, table, and image data into a ChromaDB vector store on disk.
    For images, both the image and its LLM-generated summary are embedded separately for multi-modal retrieval.
    """
    open_clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    vectorstore = Chroma(
        collection_name="summaries_and_images",
        embedding_function=open_clip_embeddings,
        persist_directory=persist_directory
    )
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
            Document(page_content=texts[i].text, metadata={id_key: doc_ids[i], "type": "text"})
            for i in range(len(texts))
        ]
        vectorstore.add_documents(summary_texts, ids=doc_ids)
        docstore.add_documents(original_text_docs, ids=doc_ids)
    # Store tables as JSON
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i], "type": "table"})
            for i, s in enumerate(table_summaries)
        ]
        original_table_docs = [
            Document(page_content=json.dumps(tables[i].dict()), metadata={id_key: table_ids[i], "type": "table"})
            for i in range(len(tables))
        ]
        vectorstore.add_documents(summary_tables, ids=table_ids)
        docstore.add_documents(original_table_docs, ids=table_ids)
    # Store images: embed both image and LLM summary separately, link by doc_id
    if image_paths and relevant_images is not None and image_summaries is not None:
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                try:
                    doc_id = str(uuid.uuid4())
                    # 1. Embed image (OpenCLIP will use the image file)
                    img_doc = Document(
                        page_content=image_path,  # This will be interpreted as an image by OpenCLIP
                        metadata={
                            id_key: doc_id,
                            "type": "image",
                            "image_path": image_path,
                            "summary": image_summaries[i],
                        }
                    )
                    # 2. Embed LLM summary (OpenCLIP will use the text)
                    summary_doc = Document(
                        page_content=image_summaries[i],
                        metadata={
                            id_key: doc_id + "_context",
                            "type": "image_summary",
                            "image_path": image_path,
                            "summary": image_summaries[i],
                        }
                    )
                    # Add both to vectorstore
                    vectorstore.add_documents([img_doc], ids=[doc_id])
                    vectorstore.add_documents([summary_doc], ids=[doc_id + "_context"])
                    # Store original image and summary in docstore
                    docstore.add_documents([
                        Document(page_content=image_path, metadata={id_key: doc_id, "type": "image", "summary": image_summaries[i]})
                    ], ids=[doc_id])
                except Exception as e:
                    print(f"Error embedding image or summary {image_path}: {e}")
            else:
                print(f"Image file not found for embedding: {image_path}")
    return UnifiedRetriever(vectorstore, docstore, id_key)

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




