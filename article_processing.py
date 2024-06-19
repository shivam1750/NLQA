import requests
from bs4 import BeautifulSoup
import torch
import os
import json

from transformers import  AutoTokenizer, AutoModel

# Load BERT tokenizer and model for embeddings
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def extract_metadata_from_article(article_content):
    # Example: Extract H1, paragraphs, and tables from article_content
    headings = [section['heading'] for section in article_content if 'heading' in section]
    paragraphs = [section['paragraph'] for section in article_content if 'paragraph' in section]
    tables = [section['table_data'] for section in article_content if 'table_data' in section]
    
    return headings, paragraphs, tables

# import numpy as np

def generate_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings


# import json
# import os

def store_article_data(article_data, filename='structured_data.json'):
    # Load existing data from the file
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new article data
    data.append(article_data)

    # Write the updated data back to the file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_and_store_articles(articles):
    for article in articles:
        headings, paragraphs, tables = extract_metadata_from_article(article['content'])
        
        for heading, paragraph in zip(headings, paragraphs):
            embeddings = generate_embeddings(paragraph)
            article_data = {
                "heading": heading,
                "paragraph": paragraph,
                "source": article['url'] , # Include the source URL
                "embeddings": embeddings

                # Add more metadata fields as needed
            }
            store_article_data(article_data)
def fetch_articles_from_website():
    # List of URLs to process
    urls = [
        'https://stanford-cs324.github.io/winter2022/lectures/introduction/',
        'https://stanford-cs324.github.io/winter2022/lectures/capabilities/',
        'https://stanford-cs324.github.io/winter2022/lectures/harms-1/',
        'https://stanford-cs324.github.io/winter2022/lectures/harms-2/',
        'https://stanford-cs324.github.io/winter2022/lectures/data/'
        # Add more URLs as needed
    ]

    all_articles = []

    for url in urls:
        print(f"Processing URL: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Target the main content div with id 'main-content'
        main_content = soup.find('div', {'id': 'main-content'})
        
        if not main_content:
            print(f"Main content not found for URL: {url}")
            continue
        
        # Extract headings and paragraphs
        content_list = []
        current_heading = None
        
        # Find all sections within main content
        sections = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div'])
        
        for section in sections:
            if section.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                current_heading = section.get_text(strip=True)
            elif section.name == 'p' and current_heading:
                paragraph = section.get_text(strip=True)
                content_list.append({
                    'heading': current_heading,
                    'paragraph': paragraph
                })
            elif section.name == 'div' and section.has_attr('class'):
                if 'table-wrapper' in section['class']:
                    # Process table content
                    table = section.find('table')
                    if table:
                        table_data = []
                        # Check if <thead> exists before accessing <th>
                        if table.find('thead'):
                            headers = [header.text.strip() for header in table.find('thead').find_all('th')]
                        else:
                            headers = []
                        
                        rows = table.find('tbody').find_all('tr')
                        for row in rows:
                            row_data = [cell.text.strip() for cell in row.find_all('td')]
                            table_data.append(dict(zip(headers, row_data)))
                        content_list.append({
                            'table_data': table_data
                        })
                elif 'MathJax_Display' in section['class']:
                    # Process MathJax content
                    mathjax_content = section.find('span', {'class': 'MathJax'}).text.strip()
                    content_list.append({
                        'type': 'mathjax',
                        'content': mathjax_content
                    })
        
        all_articles.append({'url': url, 'content': content_list})

    return all_articles
