from transformers import AutoTokenizer, AutoModelForSequenceClassification # tokenizer converts string into sequence to pass to nlp, automodel is architecture to load nlp model
import torch # argmax to extract highest sequence result
import requests # grab data/webpage
from bs4 import BeautifulSoup # traverse results from scrape, extract relevant data
import re # create regex to extract specific comments
import numpy as np
import pandas as pd
from urllib.parse import urlparse

def is_valid_yelp_url(url):
    
    # Regex to match the expected Yelp URL pattern
    yelp_url_pattern = re.compile(r"https:\/\/www\.yelp\.com\/biz\/[a-zA-Z0-9-]+")
    if not yelp_url_pattern.match(url):
        return False

    # Make a request to the URL to check if it's reachable
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return False
    except requests.exceptions.RequestException:
        return False

    return True

def get_yelp_url():
    while True:
        url = input("Please enter the Yelp URL of the restaurant you'd like to analyze. For example, https://www.yelp.com/biz/koh-lipe-toronto-2?osq=Thai: ")
        if is_valid_yelp_url(url):
            print(f"Valid Yelp URL entered: {url}")
            return url
        else:
            print("Invalid Yelp URL. Please try again.")

url = get_yelp_url()

# instantiate model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

r = requests.get(url) # grab webpage using requests lib, returns response code
soup = BeautifulSoup(r.text, "html.parser") # soup is formatted so BeautifulSoup can search
regex = re.compile(".*comment.*") # any class with "comment" in it
results = soup.find_all('p', {"class": regex}) # p tag for paragraphs, returns html tags along with text
reviews = [result.text for result in results]

# create a data frame
df = pd.DataFrame(np.array(reviews), columns=["review"])

# grab string and store in frame, still need to encode using tokenizer and pass thru model
df["review"].iloc[0]

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors="pt") # pt for pytorch!
    result = model(tokens) # outputs one-hot encoded list of scores (likeleiness of sentiment)
    return int(torch.argmax(result.logits)) + 1 # +1 bc index starts at 0 and we want value 1-5 (bad-good) of the sentiment categories

# extract review column, take every entry as x, nlp is limited to 512 tokens so grab first 512 of each review, add sentiment col
df["sentiment"] = df["review"].apply(lambda x: sentiment_score(x[:512]))
print(df)