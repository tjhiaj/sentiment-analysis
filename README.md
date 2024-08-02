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
First, I instantiate my model. AutoTokenizer is a class from Transformers that retrieves the relevant architecture based on the specified path to the pretrained model vocabulary. It creates a tokenizer that's an instance of the specified model tokenizer. Tokenizers convert strings into sequences of smaller elements called tokens (obvi) that are numerical representations the model can process. This is a crucial step in text preprocessing before passing data to an NLP model. In this case, the specified model is a version of BERT (Bidirectional Encoder Representations from Transformers) that has been trained by NLP Town for sentiment analysis on multiple languages using uncased text. AutoModelForSequenceClassification is similar to AutoTokenizer except it creates a model instead of a tokenizer. The model has a sequence classification head, meaning it takes the output of a sequence model and transforms it into a probability distribution over the possible classes.

Next, I need to grab my webpage. I'm using the Yelp page of my favourite restaurant.

> Requests is a library that helps you send HTTP requests without having to manually add query strings or form-encode your data (convert to key-value pairs separated by '&' and assigned by '=') The request returns a response object (content, encoding, status, etc.)

Then, I create soup: a BeautifulSoup object. It stores the content of the page in a nested data structure and allows me to access tags and attributes of tags, as well as methods to search for these and navigate the tree-like structure. The purpose of this project is to analyse the sentiments of various comments left on the restaurant's page, so I create a regex that will help me find any class with the string "comment" in its name.

> re.compile() compiles the specified regex pattern into a regex object so it can be used for matching.

BeautifulSoup's find_all() method takes in the p tag (paragraphs) along with the class keyword to search for and return all matches to the class (via regex) within the paragraphs. This will be a list of objects, each consisting of the tag and found text. I then extract all the text components to form a list of reviews.

The next step is to create a data frame or table to hold the reviews. This data frame will have one (numbered) column called "review". Then, I want to add a sentiment column. I want to loop through every review and calculate its sentiment score using the first 512 characters of the string (since the NLP pipeline is limited to 512 input tokens at any given time, a limitation of the BERT design).

To calculate sentiment score, I use the tokenizerModel.encode() method. It encodes the string (aka a list of tokens) into a list of token IDs based on the tokenizer's vocab to be passed to the model. The return_tensors parameter formats the outputted IDs as a PyTorch tensor since this is the default for the model, meaning the model is optimized for PyTorch rather than TensorFlow.

I pass these tokens to the model and it'll output a one-hot encoded list of probabilities for each possible sentiment score (from 1 to 5). This list is stored under the logits attribute of the returned object, and I can extract the index of the largest list value using torch.argmax(). Note that argmax() returns a PyTorch tensor of a single value, so it needs to be cast to an integer before adding 1 to it for the final sentiment score (since indices start at 0 but scores start at 1).

> Lambda functions are clean and simple, having any number of arguments but only one expression, meaning they can't capture complex logic with multiple statements. Lambda functions take on the form 'lambda arguments: expression'. They're anonymous, meaning they don't have names unless assigned to variables, so they're often used for functions that aren't meant to be reused. You'll see them used often in functional programming, such as with common functions that loop through items. In this case, apply() loops through every review in the data frame and the lambda function helps call for the sentiment_score( method each time.
