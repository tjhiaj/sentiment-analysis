from transformers import AutoTokenizer, AutoModelForSequenceClassification # tokenizer converts string into sequence to pass to nlp, automodel is architecture to load nlp model
import torch # argmax to extract highest sequence result
import requests # grab data/webpage
from bs4 import BeautifulSoup # traverse results from scrape, extract relevant data
import re # create regex to extract specific comments

# instantiate model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode("I loved this, absolutely the best", return_tensors="pt") # pt for pytorch!
result = model(tokens) # outputs one-hot encoded list of scores (likeleiness of sentiment)
# print(int(torch.argmax(result.logits)) + 1) # +1 bc index starts at 0 and we want value 1-5 (bad-good) of the sentiment categories

r = requests.get("https://www.yelp.com/biz/koh-lipe-toronto-2?osq=Thai") # grab webpage using requests lib, returns response code
soup = BeautifulSoup(r.text, "html.parser") # soup is formatted so BeautifulSoup can search
regex = re.compile(".*comment.*") # any class with "comment" in it
results = soup.find_all('p', {"class": regex}) # p tag for paragraphs, returns html tags along with text
reviews = [result.text for result in results]

# print(reviews)