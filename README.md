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
First, I instantiate my model. AutoTokenizer is a class from Transformers that retrieves the relevant architecture based on the specified path to the pretrained model vocabulary. It creates a tokenizer that's an instance of the specified model tokenizer. Tokenizers convert strings into sequences of smaller elements called tokens (obvi). This is a crucial step in text preprocessing before passing data to an NLP model. AutoModelForSequenceClassification is similar except it creates a model instead of a tokenizer. The model has a sequence classification head, meaning it takes the output of a sequence model and transforms it into a probability distribution over the possible classes.

Next, I neeed to grab my webpage. I'm using the Yelp page of my favourite restaurant.

> Requests is a library that helps you send HTTP requests without having to manually add query strings or form-encode your data (convert to key-value pairs separated by '&' and assigned by '=') The request returns a response object (content, encoding, status, etc.)

Then, I create soup: a BeautifulSoup object. It stores the content of the page in a nested data structure and allows me to access tags and attributes of tags, as well as methods to search for these and navigate the tree-like structure. The purpose of this project is to analyse the sentiments of various comments left on the restaurant's page, so I create a regex that will help me find any class with the string "comment" in its name.

> re.compile() compiles the specified regex pattern into a regex object so it can be used for matching.
