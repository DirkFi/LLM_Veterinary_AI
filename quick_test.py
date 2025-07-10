#!/usr/bin/env python3
"""
Quick test script for the fixed veterinary AI system
"""

from vet_ai_main import VeterinaryAI

def test_fixed_system():
    """Test the veterinary AI with corrected inputs."""
    
    print("🧪 Testing Fixed Veterinary AI System")
    print("=" * 60)
    
    # Initialize the AI
    try:
        vet_ai = VeterinaryAI(chroma_persist_dir="./chroma/Ears")
        print("✅ AI system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")
        return
    
    # Test case 1: Corrected query with proper image path
    print("\n" + "="*60)
    print("🧪 Test 1: Cat Health Query with Correct Image Path")
    print("="*60)
    
    result1 = vet_ai.ask(
        "tell me how can I treat my cat to make it healthy", 
        image_path="./experiment/skinny_cat.jpg"  # Corrected path
    )
    
    print("\n🎯 Final Response:")
    print(result1.get("final_answer", "No response"))
    
    # Test case 2: Text-only nutrition query
    print("\n" + "="*60)
    print("🧪 Test 2: Text-Only Nutrition Query")
    print("="*60)
    
    result2 = vet_ai.ask("My cat looks underweight and skinny. How can I help it gain healthy weight?")
    
    print("\n🎯 Final Response:")
    print(result2.get("final_answer", "No response"))
    
    # Test case 3: General health query
    print("\n" + "="*60)
    print("🧪 Test 3: General Health Query")
    print("="*60)
    
    result3 = vet_ai.ask("What are the best ways to keep my cat healthy and well-nourished?")
    
    print("\n🎯 Final Response:")
    print(result3.get("final_answer", "No response"))

if __name__ == "__main__":
    test_fixed_system()