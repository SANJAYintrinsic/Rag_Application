# RAG  - Retrieval augmented generation

- LLM  - models - Generative Pre-trained Transformer GPT 

## pdf and youtube ( Retriever )

- LLM : groq (Llama)

- Tech stack:  python, restAPI ,

## Packages:
- Langchain 
- Langraph - graph structure - nodes and edges   (conversation memory)
- Chroma - VB
- Faiss - vector library
- Hugging face
- Streamlit - web interface

- .env = file   GROQ_API_KEY="   "

# Paramater 

- Knowledge cutoff

- Tokens

- Temperature - Controls randomness (creativity) of the output.

- Max Tokens - Limits the maximum length of the response.

- Top-P (Nucleus Sampling) - Controls probability-based word selection, The model chooses words from the top 90% probability mass.

- Top-K - Limits word selection to K most probable tokens. ( The model selects the next word from the top 40 likely tokens. )

- Frequency Penalty - Reduces repeated words in output.

- Presence Penalty - Encourages new topics in responses.

- Stop Sequences  - Tells the model when to stop generating text.

- Streaming - Allows token-by-token response generation. ( AI is transforming the world... )

- Seed - Controls reproducibility. ( Same seed → same output , Useful for testing and debugging. )


## RAG - pipeline

- Documents -Text Chunking (breakdown) - ( Embeddings - Vector Database - Numbered vector)  - Retriever - LLM ( Prompt Query ) - Answer Generation

- Video - audio fetch - transcription generation - text file
