#!/usr/bin/env python3
"""
Veterinary AI Assistant - Main Application
A complete RAG-based veterinary AI assistant with multimodal capabilities.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from textbook_loading import UnifiedRetriever
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
import ollama
from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    # Core user input
    text_query: str
    image_path: Optional[str]  # Path to uploaded image, if any

    # Routing/intent
    query_type : str # currently being divide into 'emergency'/'Q&A'/'irrelevant'.
    
    # Q&A path
    refined_query: Optional[str]
    sub_queries: Optional[List[str]]
    current_sub_query: Optional[str]
    retrieved_docs: Optional[List[Dict[str, Any]]]  # Results from retrieval
    reranked_docs: Optional[List[Dict[str, Any]]]   # After rerank step

    # Feedback loop
    followup_questions: Optional[List[str]]
    user_responses: Optional[List[str]]
    loop_count: int

    # Answer generation
    generated_answer: Optional[str]
    hallucination_check: Optional[bool]
    answer_sufficient: Optional[bool]

    # Emergency path
    emergency_instructions: Optional[str]
    emergency_retrieved_docs: Optional[List[Dict[str, Any]]]

    # Web search
    web_search_results: Optional[List[Dict[str, Any]]]

    # Final output
    final_answer: Optional[str]

    # Misc/trace/debug
    path_taken: Optional[List[str]]
    error: Optional[str]

class VeterinaryAI:
    def __init__(self, chroma_persist_dir="./chroma/Ears"):
        """
        Initialize the Veterinary AI Assistant.
        
        Args:
            chroma_persist_dir (str): Path to the ChromaDB persistence directory
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.retriever = self._init_retriever()
        self.graph = self._create_graph()
        
    def _init_retriever(self):
        """Initialize the unified retriever for multimodal search."""
        open_clip_embeddings = OpenCLIPEmbeddings(
            model_name="ViT-g-14", 
            checkpoint="laion2b_s34b_b88k"
        )

        # Vectorstore for summaries (for similarity search)
        vectorstore = Chroma(
            collection_name="summaries_and_images",
            persist_directory=self.chroma_persist_dir,
            embedding_function=open_clip_embeddings
        )
        
        # Persistent docstore for originals (all modalities)
        docstore = Chroma(
            collection_name="originals",
            persist_directory=self.chroma_persist_dir,
            embedding_function=open_clip_embeddings
        )

        return UnifiedRetriever(vectorstore, docstore, id_key="doc_id")
    
    def _get_image_summary(self, image_path):
        """Get a detailed summary of an image using vision model."""
        if not image_path or not os.path.exists(image_path):
            return ""
            
        prompt = """From a feline veterinary standpoint, provide a highly detailed and objective 
                    description of the image. Focus on all observable elements, actions, 
                    objects, subjects, their attributes (e.g., color, size, texture), 
                    their spatial relationships, and any discernible context or implied scene. 
                    Also focus on all possible health issues.
                    Describe any text present in the image. This description must be exhaustive 
                    and purely factual, capturing every significant visual detail to serve as a 
                    comprehensive textual representation for further analysis by another AI model. 
                    If the image is entirely irrelevant or contains no discernible subject, 
                    state "No relevant visual information."""
        
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
        
        try:
            response = ollama.chat(
                model="minicpm-v:8b",
                messages=messages,
                options={"temperature": 0.2}
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error getting image summary: {e}")
            return ""

    def _create_graph(self):
        """Create the complete LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add all nodes
        workflow.add_node("query_handler", self._query_handler)
        workflow.add_node("query_refinement", self._query_refinement)
        workflow.add_node("query_decomposition", self._query_decomposition)
        workflow.add_node("contextual_retrieval", self._contextual_retrieval)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("thinking", self._thinking)
        workflow.add_node("answer_generation", self._answer_generation)
        workflow.add_node("hallucination_check", self._hallucination_check)
        workflow.add_node("emergency_handler", self._emergency_handler)
        workflow.add_node("irrelevant_handler", self._irrelevant_handler)
        workflow.add_node("finalize_answer", self._finalize_answer)
        workflow.add_node("regenerate_answer", self._regenerate_answer)
        
        # Set entry point
        workflow.set_entry_point("query_handler")
        
        # Add edges and conditional routing
        workflow.add_conditional_edges(
            "query_handler",
            self._route_after_query_handler,
            {
                "emergency_handler": "emergency_handler",
                "query_refinement": "query_refinement", 
                "irrelevant_handler": "irrelevant_handler"
            }
        )
        
        # Q&A path edges
        workflow.add_edge("query_refinement", "query_decomposition")
        workflow.add_edge("query_decomposition", "contextual_retrieval")
        workflow.add_edge("contextual_retrieval", "rerank")
        workflow.add_edge("rerank", "thinking")
        workflow.add_edge("thinking", "answer_generation")
        workflow.add_edge("answer_generation", "hallucination_check")
        
        # Conditional routing after hallucination check
        workflow.add_conditional_edges(
            "hallucination_check",
            self._route_after_hallucination_check,
            {
                "finalize_answer": "finalize_answer",
                "regenerate_answer": "regenerate_answer"
            }
        )
        
        # Terminal edges
        workflow.add_edge("emergency_handler", END)
        workflow.add_edge("irrelevant_handler", END)
        workflow.add_edge("finalize_answer", END)
        workflow.add_edge("regenerate_answer", END)
        
        return workflow.compile()

    def _query_handler(self, state):
        """Classify the query type."""
        text_query = state.get("text_query", "")
        image_path = state.get("image_path", None)

        prompt = (
            "You are a domain classifier for a veterinary assistant. "
            "If an image is provided, understand the image from veterinary point of view."
            "A user query is the combination of text query and image(if there is). "
            "Then, classify the user query into one of three categories:\n"
            "1. 'emergency' — If the user query is about a veterinary emergency (e.g., mass bleeding, serious bone fracture, unconsciousness, severe breathing difficulty, or other life-threatening situations).\n"
            "2. 'Q&A' — If the user query is about is about general veterinary questions, symptom checks, or non-emergency animal health issues.\n\n"
            "3. 'irrelevant' — If the user query is NOT about veterinary, animal health, pet care, etc.\n"
            "Your response must be exactly one of: 'irrelevant', 'emergency', or 'Q&A'. Do not explain your answer or add anything else.\n\n"
            f"User input: {text_query}\n"
        )

        messages = [{
            "role": "user",
            "content": prompt,
            "images": []
        }]

        if image_path and os.path.exists(image_path):
            messages[0]["images"].append(image_path)

        try:
            response = ollama.chat(
                model="minicpm-v:8b",
                messages=messages,
                options={"temperature": 0.2}
            )
            result = response['message']['content'].strip().lower()
            if result not in ['irrelevant', 'emergency', 'q&a']:
                result = 'irrelevant'
            
            return {"query_type": result}
        except Exception as e:
            print(f"Error in query handler: {e}")
            return {"query_type": "q&a"}  # Default to Q&A if error

    def _route_after_query_handler(self, state):
        """Route after query classification."""
        query_type = state.get("query_type", "")
        if query_type == "emergency":
            return "emergency_handler"
        elif query_type == "q&a":
            return "query_refinement"
        else:
            return "irrelevant_handler"

    def _query_refinement(self, state):
        """Refine and expand the user query."""
        text_query = state.get("text_query", "")
        image_path = state.get("image_path", None)
        image_summary = self._get_image_summary(image_path) if image_path else ""

        if image_summary:
            prompt = (
                "You are a veterinary assistant AI. Your task is to rewrite and expand the user's query for a veterinary knowledge base search. "
                "Use the image description to add context, but avoid making assumptions about the specific diagnosis. "
                "Frame the refined query in an open-ended, unbiased way. "
                "Output ONLY one single, context-rich, and unbiased query as a paragraph, and nothing else.\n\n"
                f"User query: {text_query}\n"
                f"Image description: {image_summary}\n"
                "Refined query:"
            )
        else:
            prompt = (
                "You are a veterinary assistant AI. Your task is to rewrite and expand the user's query for a veterinary knowledge base search. "
                "Consider possible causes, diagnostic considerations, anything that would be helpful. "
                "Output ONLY one single, context-rich query as a paragraph, and nothing else.\n\n"
                f"User query: {text_query}\n"
                "Refined query:"
            )

        try:
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            return {"refined_query": response['message']['content']}
        except Exception as e:
            print(f"Error in query refinement: {e}")
            return {"refined_query": text_query}

    def _query_decomposition(self, state):
        """Decompose complex query into sub-queries."""
        refined_query = state.get('refined_query', '')
        
        # For now, return the refined query as single sub-query
        # This can be expanded with more sophisticated decomposition
        return {"sub_queries": [refined_query]}

    def _contextual_retrieval(self, state):
        """Retrieve relevant documents."""
        sub_queries = state.get('sub_queries', [])
        seen_doc_ids = set()
        unique_docs = []

        for query in sub_queries:
            try:
                results = self.retriever.retrieve_multi_modal(query, k=5)
                for res in results:
                    doc_id = res.get('doc_id') or res.get('summary_metadata', {}).get('doc_id')
                    if doc_id and doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        unique_docs.append(res)
            except Exception as e:
                print(f"Error in retrieval for query '{query}': {e}")
                continue

        return {"retrieved_docs": unique_docs}

    def _rerank(self, state):
        """Rerank retrieved documents."""
        # For now, return documents as-is
        # This can be expanded with more sophisticated reranking
        retrieved_docs = state.get("retrieved_docs", [])
        return {"reranked_docs": retrieved_docs}

    def _thinking(self, state):
        """Analyze query and documents."""
        text_query = state.get("text_query", "")
        refined_query = state.get("refined_query", "")
        reranked_docs = state.get("reranked_docs", [])
        
        # Prepare context
        context_pieces = []
        for doc in reranked_docs[:5]:  # Top 5 docs
            modality = doc.get("modality") or (doc.get("original_metadata") or {}).get("type")
            if modality == "text":
                context_pieces.append(f"[TEXT] {doc.get('summary', '')}")
            elif modality in ["image", "image_summary"]:
                context_pieces.append(f"[IMAGE] {doc.get('summary', '')}")
        
        context = "\n".join(context_pieces)
        
        return {
            "thinking_analysis": f"Analyzing query: {text_query}",
            "context_for_answer": context
        }

    def _answer_generation(self, state):
        """Generate the final answer."""
        text_query = state.get("text_query", "")
        context_for_answer = state.get("context_for_answer", "")
        
        prompt = f"""
        You are a knowledgeable veterinary assistant providing helpful information to pet owners.
        
        IMPORTANT GUIDELINES:
        1. Always emphasize consulting with a veterinarian for proper diagnosis and treatment
        2. Provide factual, evidence-based information
        3. Include relevant warnings about emergency situations
        4. Be empathetic and supportive in tone
        5. Reference specific information from the retrieved documents
        
        User Query: {text_query}
        
        Retrieved Information:
        {context_for_answer}
        
        Please provide a comprehensive, helpful response that addresses the user's concern.
        """
        
        try:
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4}
            )
            return {"generated_answer": response['message']['content']}
        except Exception as e:
            print(f"Error in answer generation: {e}")
            return {"generated_answer": "I apologize, but I'm having trouble generating a response right now. Please consult with a veterinarian for your pet's health concerns."}

    def _hallucination_check(self, state):
        """Check for hallucinations in the generated answer."""
        # For now, assume all answers pass
        # This can be expanded with more sophisticated checking
        return {"hallucination_check": True}

    def _route_after_hallucination_check(self, state):
        """Route after hallucination check."""
        hallucination_check = state.get("hallucination_check", False)
        return "finalize_answer" if hallucination_check else "regenerate_answer"

    def _finalize_answer(self, state):
        """Finalize the answer."""
        return {
            "final_answer": state.get("generated_answer", ""),
            "path_taken": state.get("path_taken", []) + ["Q&A_completed"]
        }

    def _regenerate_answer(self, state):
        """Regenerate answer with warning."""
        original_answer = state.get("generated_answer", "")
        warning = "\n\n⚠️ Please note: Some information may need verification. Always consult with a veterinarian for accurate diagnosis and treatment."
        
        return {
            "final_answer": original_answer + warning,
            "path_taken": state.get("path_taken", []) + ["regenerated_with_warning"]
        }

    def _emergency_handler(self, state):
        """Handle emergency queries."""
        text_query = state.get("text_query", "")
        
        emergency_response = f"""
        ⚠️ EMERGENCY: Contact your veterinarian or emergency animal hospital IMMEDIATELY
        
        Your situation: {text_query}
        
        IMMEDIATE ACTIONS:
        1. Stay calm and act quickly
        2. Call your veterinarian or emergency animal hospital now
        3. If your pet is unconscious, ensure the airway is clear
        4. Apply gentle pressure to any bleeding wounds with clean cloth
        5. Keep your pet warm and quiet
        6. Transport carefully to avoid further injury
        
        Emergency numbers:
        - Your veterinarian: [Contact your vet]
        - Emergency animal hospital: [Find nearest emergency clinic]
        
        This is a medical emergency requiring immediate professional veterinary care.
        """
        
        return {"final_answer": emergency_response}

    def _irrelevant_handler(self, state):
        """Handle irrelevant queries."""
        text_query = state.get("text_query", "")
        
        response = f"""
        I'm a veterinary AI assistant designed to help with animal health and pet care questions.
        
        Your query: "{text_query}"
        
        This doesn't appear to be related to veterinary medicine, animal health, or pet care.
        
        I can help you with:
        • Animal health symptoms and concerns
        • Pet care advice
        • Veterinary procedures and treatments
        • Emergency animal care
        • General pet wellness questions
        
        Please feel free to ask me anything related to animal health or pet care!
        """
        
        return {"final_answer": response}

    def ask(self, query, image_path=None):
        """
        Process a user query and return the AI's response.
        
        Args:
            query (str): The user's question
            image_path (str, optional): Path to an image file
            
        Returns:
            dict: The complete response from the AI system
        """
        initial_state = {
            "text_query": query,
            "image_path": image_path,
            "loop_count": 0,
            "path_taken": []
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "final_answer": "I apologize, but I encountered an error processing your query. Please try again or consult with a veterinarian.",
                "error": str(e)
            }

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Veterinary AI Assistant")
    parser.add_argument("--query", "-q", type=str, help="Your veterinary question")
    parser.add_argument("--image", "-i", type=str, help="Path to image file")
    parser.add_argument("--chroma-dir", type=str, default="./chroma/Ears", 
                       help="Path to ChromaDB directory")
    parser.add_argument("--interactive", "-I", action="store_true", 
                       help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the AI
    print("🐾 Initializing Veterinary AI Assistant...")
    try:
        vet_ai = VeterinaryAI(chroma_persist_dir=args.chroma_dir)
        print("✅ AI Assistant ready!")
    except Exception as e:
        print(f"❌ Error initializing AI: {e}")
        return 1
    
    if args.interactive:
        # Interactive mode
        print("\n🎯 Interactive Mode - Type 'quit' to exit")
        print("You can ask questions about pet health, symptoms, or veterinary care.")
        print("To include an image, type 'image:path/to/image.jpg' on a separate line.\n")
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                image_path = None
                if query.startswith('image:'):
                    image_path = query[6:].strip()
                    query = input("Question with image: ").strip()
                
                if not query:
                    continue
                
                print("\n🔍 Processing your question...")
                result = vet_ai.ask(query, image_path)
                
                print("\n🎯 Response:")
                print(result.get("final_answer", "No response generated"))
                
                # Show additional info if available
                if "query_type" in result:
                    print(f"\n📂 Query type: {result['query_type']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.query:
        # Single query mode
        print(f"\n🔍 Processing: {args.query}")
        if args.image:
            print(f"📸 With image: {args.image}")
        
        result = vet_ai.ask(args.query, args.image)
        
        print("\n🎯 Response:")
        print(result.get("final_answer", "No response generated"))
        
        if "query_type" in result:
            print(f"\n📂 Query type: {result['query_type']}")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())