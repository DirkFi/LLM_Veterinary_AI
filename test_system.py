#!/usr/bin/env python3
"""
Test script for the Veterinary AI System
"""

import os
import sys
from vet_ai_main import VeterinaryAI

def test_veterinary_ai():
    """Test the complete veterinary AI system."""
    
    print("🐾 Testing Veterinary AI System")
    print("=" * 50)
    
    # Initialize the AI
    try:
        vet_ai = VeterinaryAI(chroma_persist_dir="./chroma/Ears")
        print("✅ AI system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "Regular Q&A Query",
            "query": "My cat has been scratching its ear a lot and I see dark stuff inside. What could this be?",
            "expected_type": "q&a"
        },
        {
            "name": "Emergency Query", 
            "query": "My cat is bleeding heavily from a deep wound and seems unconscious!",
            "expected_type": "emergency"
        },
        {
            "name": "Irrelevant Query",
            "query": "How do I fix my car's engine?",
            "expected_type": "irrelevant"
        },
        {
            "name": "Simple Health Query",
            "query": "What vaccines does my cat need?",
            "expected_type": "q&a"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print("-" * 30)
        
        try:
            result = vet_ai.ask(test_case['query'])
            
            # Check if we got a response
            if 'final_answer' in result and result['final_answer']:
                print("✅ Response generated successfully")
                
                # Check query classification
                query_type = result.get('query_type', 'unknown')
                expected_type = test_case['expected_type']
                
                if query_type == expected_type:
                    print(f"✅ Query correctly classified as: {query_type}")
                    test_result = "PASS"
                else:
                    print(f"⚠️ Query classification mismatch: expected {expected_type}, got {query_type}")
                    test_result = "PARTIAL"
                
                # Show first 100 chars of response
                response_preview = result['final_answer'][:100] + "..." if len(result['final_answer']) > 100 else result['final_answer']
                print(f"Response preview: {response_preview}")
                
            else:
                print("❌ No response generated")
                test_result = "FAIL"
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
            test_result = "FAIL"
        
        results.append({
            'test': test_case['name'],
            'result': test_result
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['result'] == 'PASS')
    partial = sum(1 for r in results if r['result'] == 'PARTIAL')
    failed = sum(1 for r in results if r['result'] == 'FAIL')
    
    for result in results:
        status_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}[result['result']]
        print(f"{status_emoji} {result['test']}: {result['result']}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Partial: {partial}")
    print(f"Failed: {failed}")
    
    success_rate = (passed + partial * 0.5) / len(results) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    return success_rate >= 75  # Consider 75% success rate as good

def test_with_image():
    """Test the system with an image if available."""
    
    print("\n🖼️ Testing with image...")
    
    # Check if test image exists
    image_path = "./experiment/cat_ear_problem.jpeg"
    if not os.path.exists(image_path):
        print(f"⚠️ Test image not found at {image_path}")
        return
    
    try:
        vet_ai = VeterinaryAI(chroma_persist_dir="./chroma/Ears")
        
        result = vet_ai.ask(
            "What's wrong with my cat's ear?", 
            image_path=image_path
        )
        
        if 'final_answer' in result and result['final_answer']:
            print("✅ Image processing successful")
            print(f"Query type: {result.get('query_type', 'unknown')}")
            print(f"Response preview: {result['final_answer'][:150]}...")
        else:
            print("❌ Image processing failed")
            
    except Exception as e:
        print(f"❌ Error testing with image: {e}")

if __name__ == "__main__":
    print("🧪 Running Veterinary AI System Tests")
    print("This will test the complete RAG pipeline including:")
    print("• Query classification")
    print("• Document retrieval")
    print("• Answer generation")
    print("• Emergency handling")
    print("• Irrelevant query handling")
    print()
    
    # Run main tests
    success = test_veterinary_ai()
    
    # Test with image if available
    test_with_image()
    
    print("\n🎯 Test Summary:")
    if success:
        print("✅ Overall system test: PASSED")
        print("The veterinary AI system is working correctly!")
    else:
        print("⚠️ Overall system test: NEEDS IMPROVEMENT")
        print("Some components may need adjustment.")
    
    print("\n💡 Next steps:")
    print("1. Run the main application: python vet_ai_main.py --interactive")
    print("2. Test with specific queries: python vet_ai_main.py -q 'your question'")
    print("3. Test with images: python vet_ai_main.py -q 'your question' -i 'path/to/image.jpg'")