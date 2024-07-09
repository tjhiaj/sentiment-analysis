# Sentiment Analysis

A sentiment calculator that leverages NLP. Applies Hugging Face Transformers and a BERT neural network.


## Features

- Displays the top 10 reviews of a restaurant and calculates their associated sentiments
- Sentiment scale from 1 (negative) to 5 (positive)
- Scrapes Yelp for restaurant reviews


## Tech Stack

**Language:** Python

**Frameworks:** Hugging Face Transformers, BERT, PyTorch, NumPy, Pandas

**Dev Tools:** BeautifulSoup, Requests, re


## How I Built It
First, I instantiate my model. AutoTokenizer is a class from Transformers that retrieves the relevant architecture based on the specified path to the pretrained model vocabulary. It creates a tokenizer that's an instance of the specified model tokenizer. AutoModelForSequenceClassification is similar except it creates a model instead ofa  tokenizer. The model has a sequence classification head, meaning
