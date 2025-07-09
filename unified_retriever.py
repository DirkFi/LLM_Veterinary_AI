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