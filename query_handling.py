import spacy
from transformers import  AutoTokenizer, AutoModelForCausalLM, AutoModel
import faiss
import json
import torch
import numpy as np

# Load BERT tokenizer and model for embeddings
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")


# Load the spacy model for NER
nlp = spacy.load("en_core_web_sm")
# Load the tokenizer and model for Bloom from Hugging Face model hub
bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
bloom_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
def parse_query(query):
    tokens = tokenize_query(query)
    entities = perform_ner(query)
    parsed_query = {
        'tokens': tokens,
        'entities': entities
    }
    return parsed_query

def load_structured_data(filename='structured_data.json'):
    # Load structured data from JSON file with UTF-8 encoding
    with open(filename, 'r', encoding='utf-8') as infile:
        structured_data = json.load(infile)
    return structured_data

def tokenize_query(query):
    # Tokenize the query
    tokens = query.split()  # Basic tokenization
    return tokens

def perform_ner(query):
    # Perform named entity recognition (NER) on the query
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
def generate_answer(parsed_query, structured_data):
    tokens = parsed_query['tokens']
    # entities = parsed_query['entities']
    
    # Prepare input text for the model
    input_text = ' '.join(tokens)
    
    # Generate answer using the Bloom model
    inputs = bloom_tokenizer(input_text, return_tensors="pt")
    outputs = bloom_model.generate(inputs.input_ids,  max_length=len(inputs.input_ids[0]) + 100,num_beams=5, num_return_sequences=1, early_stopping=True, no_repeat_ngram_size=2, repetition_penalty=2.0)
    
    # Decode the generated answer
    bloom_answer = bloom_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure the answer ends with a complete sentence
    if bloom_answer:
        if bloom_answer[-1] not in '.!?':
            last_punctuation_index = max(bloom_answer.rfind('.'), bloom_answer.rfind('!'), bloom_answer.rfind('?'))
            if last_punctuation_index != -1:
                bloom_answer = bloom_answer[:last_punctuation_index + 1]
            else:
                bloom_answer += '.'    
    
    return bloom_answer  

# faiss indexing
# import faiss

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Assuming embeddings are numpy arrays
    index.add(embeddings)
    return index

def search_similar_articles(query_embeddings, faiss_index, k=5):
    # Ensure query_embeddings is a 1D array and of type np.float32
    query_embeddings = query_embeddings.reshape(1, -1).astype(np.float32)
    
    # Perform a search using the Faiss index
    distances, indices = faiss_index.search(query_embeddings, k)
    return distances, indices

# import numpy as np

def generate_embeddings2(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Convert to NumPy array
    return embeddings

def query_articles(query_text):
    query_embeddings = generate_embeddings2(query_text)
    
    # Load embeddings from structured_data.json
    all_embeddings = []
    with open('structured_data.json', 'r', encoding='utf-8') as f:
        structured_data = json.load(f)
        
    for article_data in structured_data:
        embeddings = np.array(article_data['embeddings'])  # Ensure embeddings are converted to NumPy array
        all_embeddings.append(embeddings)
    
    # Build Faiss index
    embeddings_np = np.stack(all_embeddings)
    faiss_index = build_faiss_index(embeddings_np)
    
    # Search for similar articles based on query embeddings
    distances, indices = search_similar_articles(query_embeddings, faiss_index)
    
    # Fetch and process top-k similar articles
    top_k_articles = []
    for idx in indices[0]:
        if idx < len(structured_data):
            article_data = structured_data[idx]
            top_k_articles.append(article_data)
    
    return top_k_articles





def generate_answer_with_metadata(parsed_query, structured_data):
    relevant_articles = query_articles(' '.join(parsed_query['tokens']))
    relevant_paragraphs = [article['paragraph'] for article in relevant_articles]

    # Use a set to keep track of unique URLs
    relevant_sources = set(article['source'] for article in relevant_articles)

    # Adjust the query based on relevant paragraphs or other metadata
    modified_query = ' '.join(parsed_query['tokens']) + ' '.join(relevant_paragraphs)
     # Generate answer using the modified query
    generated_answer = generate_answer({'tokens': modified_query.split()}, structured_data)
    
    # Combine the generated answer with references
    answer_with_references = generated_answer + "\n\nReferences:\n" + "\n".join(relevant_sources)
    
    return answer_with_references
    
