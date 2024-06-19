import spacy
import json
import requests
from bs4 import BeautifulSoup
import torch
import numpy as np
import faiss
import os
# from transformers import T5Tokenizer, T5ForConditionalGeneration


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel,T5Tokenizer, T5ForConditionalGeneration

from article_processing import fetch_articles_from_website, process_and_store_articles
from query_handling import generate_answer_with_metadata, load_structured_data, parse_query
# from summary_generation import add_to_conversation_history, generate_conversation_summary

# Load the spacy model for NER
nlp = spacy.load("en_core_web_sm")
# Load the tokenizer and model for Bloom from Hugging Face model hub
bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
bloom_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# # Load the tokenizer and model for Bloom from local storage
# bloom_tokenizer = AutoTokenizer.from_pretrained("local_bloom_tokenizer")
# bloom_model = AutoModelForCausalLM.from_pretrained("local_bloom_model")

# Load BERT tokenizer and model for embeddings
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Save the tokenizer and model locally
# bert_tokenizer.save_pretrained("local_bert_tokenizer")
# bert_model.save_pretrained("local_bert_model")
# Load the tokenizer and model for summarization
summarization_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarization_model = T5ForConditionalGeneration.from_pretrained('t5-small')

conversation_history = []
from transformers import T5Tokenizer, T5ForConditionalGeneration

def add_to_conversation_history(user_input, system_response):
    conversation_history.append({
        'user': user_input,
        'system': system_response
    })


# Load the tokenizer and model for summarization
summarization_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarization_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def generate_conversation_summary(conversation_history):
    # Combine the conversation history into a single string
    conversation_text = " ".join(
        [f"User: {turn['user']} System: {turn['system']}" for turn in conversation_history]
    )

    # Prepare the input for the summarization model
    inputs = summarization_tokenizer.encode("summarize: " + conversation_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = summarization_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True, no_repeat_ngram_size=2, repetition_penalty=1.5 )
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def main():
    structured_data_filename = 'structured_data.json'

    # Check if structured_data.json exists
    # if False:
    if os.path.exists(structured_data_filename):
        print("Loading existing structured data...")
        structured_data = load_structured_data()
    else:
        print("Fetching articles from the website and processing them...")
        # Example: Fetch articles from a source (e.g., website)
        articles = fetch_articles_from_website()  # Implement as per your data source
        process_and_store_articles(articles)
        structured_data = load_structured_data()
    
    
    while True:
        # Example query processing (replace with user input handling)
        query = input("Enter your query (or type 'exit' to quit, 'summary' for session summary): ")
        
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        elif query.lower() == 'summary':
            summary = generate_conversation_summary(conversation_history)
            print(f"Conversation Summary: {summary}")
            continue
        
        parsed_query = parse_query(query)
        print(f"Parsed Query: {parsed_query}")
        
        answer = generate_answer_with_metadata(parsed_query, structured_data)
        print(f"Answer: {answer}")
        
        add_to_conversation_history(query, answer)


if __name__ == "__main__":
    main()

