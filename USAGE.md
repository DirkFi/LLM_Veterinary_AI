# 🐾 Veterinary AI Assistant - Usage Guide

## 🎯 Overview

The Veterinary AI Assistant is a complete RAG (Retrieval-Augmented Generation) system that provides intelligent responses to veterinary and pet care questions. It supports both text and image inputs and can handle different types of queries including emergencies.

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- Python 3.8+
- All dependencies installed (see `requirements.txt`)
- Ollama installed with the following models:
  - `llama3.2:3b` (for text processing)
  - `minicpm-v:8b` (for image processing)
- ChromaDB vector database populated (run `ingestion.ipynb` first)

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Test the system
python test_system.py
```

### 3. Basic Usage

#### Interactive Mode (Recommended)
```bash
python vet_ai_main.py --interactive
```

This starts an interactive chat where you can:
- Ask questions about pet health
- Get emergency guidance
- Upload images for analysis

#### Single Query Mode
```bash
# Text-only query
python vet_ai_main.py -q "My cat is scratching its ear a lot"

# Query with image
python vet_ai_main.py -q "What's wrong with my cat?" -i "path/to/image.jpg"
```

## 🔧 System Components

### 1. Query Classification
The system automatically classifies queries into:
- **🚨 Emergency**: Urgent medical situations requiring immediate veterinary attention
- **❓ Q&A**: General health questions and symptoms
- **🚫 Irrelevant**: Non-veterinary questions (politely redirected)

### 2. Multimodal Processing
- **Text Analysis**: Natural language processing for symptom description
- **Image Analysis**: Computer vision for visual symptoms and conditions
- **Combined Analysis**: Integrated text+image understanding

### 3. Knowledge Retrieval
- **Vector Search**: Semantic similarity search through veterinary textbooks
- **Multimodal Retrieval**: Retrieves both text and image content
- **Contextual Ranking**: Reranks results based on relevance to query

### 4. Answer Generation
- **Evidence-Based**: Responses grounded in retrieved veterinary knowledge
- **Safety-First**: Always recommends professional veterinary consultation
- **Empathetic**: Supportive tone appropriate for pet owners

## 📝 Example Interactions

### Regular Health Question
```
🤔 Your question: My cat has been scratching its ear and I see dark discharge

🎯 Response:
Based on the symptoms you've described, your cat may be experiencing an ear infection or ear mite infestation. The dark discharge and excessive scratching are common signs of:

1. **Ear Mites**: Tiny parasites that cause dark, coffee-ground-like discharge
2. **Bacterial Otitis**: Infection causing inflammation and discharge
3. **Yeast Infection**: Can develop from excessive moisture or antibiotics

**Immediate Care:**
- Don't attempt to clean deep into the ear canal
- Prevent your cat from scratching excessively
- Monitor for worsening symptoms

**⚠️ Important:** Please schedule an appointment with your veterinarian for proper diagnosis and treatment. They can examine the ear, identify the specific cause, and prescribe appropriate medication.

📂 Query type: q&a
```

### Emergency Situation
```
🤔 Your question: My cat is bleeding heavily and won't wake up

🎯 Response:
⚠️ EMERGENCY: Contact your veterinarian or emergency animal hospital IMMEDIATELY

Your situation: My cat is bleeding heavily and won't wake up

IMMEDIATE ACTIONS:
1. Stay calm and act quickly
2. Call your veterinarian or emergency animal hospital now
3. If your pet is unconscious, ensure the airway is clear
4. Apply gentle pressure to any bleeding wounds with clean cloth
5. Keep your pet warm and quiet
6. Transport carefully to avoid further injury

This is a medical emergency requiring immediate professional veterinary care.

📂 Query type: emergency
```

### With Image Analysis
```
🤔 Your question: What's wrong with my cat's ear?
📸 With image: cat_ear_problem.jpeg

🎯 Response:
Based on the image and your concern, I can see what appears to be redness and discharge in your cat's ear canal. This visual presentation, combined with the brownish debris visible, suggests possible:

1. **Ear Mite Infestation**: The dark, waxy discharge is characteristic of ear mites
2. **Bacterial Infection**: Secondary infection from scratching or mites
3. **Inflammatory Response**: Redness indicating irritation or infection

**What I observe in the image:**
- Reddish-brown discharge in the ear canal
- Inflammation of the ear tissues
- Possible scratching wounds around the ear

**Recommended Actions:**
- Schedule a veterinary examination within 24-48 hours
- Prevent excessive scratching (soft cone collar if needed)
- Don't attempt to clean deeply into the ear

**⚠️ Veterinary Consultation Required:** A proper diagnosis requires microscopic examination of the ear discharge and professional assessment.

📂 Query type: q&a
```

## 🛠️ Advanced Configuration

### Custom ChromaDB Location
```bash
python vet_ai_main.py --chroma-dir "/path/to/your/chroma/db" --interactive
```

### Command Line Options
```bash
python vet_ai_main.py --help
```

Options:
- `--query, -q`: Your veterinary question
- `--image, -i`: Path to image file
- `--chroma-dir`: Path to ChromaDB directory
- `--interactive, -I`: Start interactive mode

## 🔍 System Architecture

```
User Input (Text + Image)
           ↓
    Query Classification
           ↓
   ┌─────────────────────┐
   │                     │
Emergency    Q&A       Irrelevant
   │         │            │
   │    Query Refinement  │
   │         │            │
   │    Query Decomposition
   │         │            │
   │    Document Retrieval
   │         │            │
   │    Result Reranking  │
   │         │            │
   │    Answer Generation │
   │         │            │
   │    Hallucination Check
   │         │            │
   └─────────────────────┘
           ↓
    Final Response
```

## 🚨 Safety Features

1. **Emergency Detection**: Automatically identifies urgent situations
2. **Professional Recommendation**: Always suggests veterinary consultation
3. **Hallucination Prevention**: Checks responses against source material
4. **Scope Limitation**: Only responds to veterinary-related queries

## 📚 Knowledge Base

The system uses:
- **Veterinary Textbooks**: Professional veterinary medicine content
- **Symptom Descriptions**: Detailed symptom analysis
- **Treatment Procedures**: Step-by-step care instructions
- **Emergency Protocols**: First aid and emergency procedures
- **Visual References**: Anatomical diagrams and condition images

## 🐛 Troubleshooting

### Common Issues

1. **"No response generated"**
   - Check ChromaDB connection
   - Verify Ollama models are running
   - Ensure vector database is populated

2. **"Error in query handler"**
   - Restart Ollama service
   - Check model availability: `ollama list`
   - Verify model names match configuration

3. **Slow responses**
   - Normal for first query (model loading)
   - Consider GPU acceleration for faster processing
   - Reduce context window if needed

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB for models and database
- **Network**: Internet connection for initial model download

## 🔄 Updating the Knowledge Base

To add new veterinary content:

1. Place new PDF files in the `data/` directory
2. Update the ingestion notebook with new file paths
3. Run the ingestion process: `jupyter notebook ingestion.ipynb`
4. The system will automatically use the updated knowledge base

## 📞 Support

For technical issues:
- Check the troubleshooting section above
- Review system logs for error messages
- Ensure all dependencies are properly installed

**Remember**: This is an AI assistant tool. Always consult with qualified veterinary professionals for actual medical diagnosis and treatment of animals.

## 🎯 Next Steps

1. **Try the interactive mode**: `python vet_ai_main.py --interactive`
2. **Test with your own images**: Use the `-i` flag to upload pet photos
3. **Explore different query types**: Try health questions, emergency scenarios, and general pet care
4. **Customize for your needs**: Modify the knowledge base or adjust response styles

Enjoy using your Veterinary AI Assistant! 🐾