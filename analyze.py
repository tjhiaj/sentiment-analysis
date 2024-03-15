from transformers import AutoTokenizer, AutoModelForSequenceClassification # tokenizer converts string into sequence to pass to nlp, automodel is architecture to load nlp model
import torch # argmax to extract highest sequence result
import requests # grab data/webpage
from bs4 import BeautifulSoup # traverse results from scrape, extract relevant data
import re # create regex to extract specific comments