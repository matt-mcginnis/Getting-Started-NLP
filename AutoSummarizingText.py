import requests
import lxml.html as LH
from lxml.html import fromstring
from bs4 import BeautifulSoup

### Downloading an Article ###

# Define URL for the article we want to analyze
articleURL = "https://theconversation.com/declassified-cold-war-code-breaking-manual-has-lessons-for-solving-impossible-puzzles-161595"

# Make the request to the URL above and get the content
page = requests.get(articleURL).content.decode("utf8", "ignore")

# Instantiate new Beautiful Soup object for the page and filter text
soup = BeautifulSoup(page, "lxml")
#print(soup.find("article"))
#print(soup.find("article").text)

# Need to use b in replace to convert to bytes-like object instead of str
text = " ".join(map(lambda p: p.text, soup.find_all("article")))
text.encode("ascii", errors="replace").replace(b"?", b" ")
#print(text)

# Combining logic to scrape articles on a web page into a function
def getArticleText(articleURL):
    page = requests.get(articleURL).content.decode("utf8", "ignore")
    soup = BeautifulSoup(page, "lxml")
    text = " ".join(map(lambda p: p.text.replace("\n", ""), soup.find_all("p")))
    return text #text.encode("ascii", errors="replace").replace(b"?", b" ")

text = getArticleText(articleURL)
#print(text)

### Preprocessing Article Text ###
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Tokenize Sentences from Article
sents = sent_tokenize(text)
#print(sents)

# Tokenize Words from Article
word_sent = word_tokenize(text.lower())
#print(word_sent)

# Define Set of Stopwords
_stopwords = set(stopwords.words("english") + list(punctuation))
#print(_stopwords)

# Get the List of Words in the Article, Not Including Stopwords
word_sent = [word for word in word_sent if word not in _stopwords]
#print(word_sent)

### Extracting a Summary ###

# Generate a frequency distribution for the list of words
from nltk.probability import FreqDist
freq = FreqDist(word_sent)
#print(freq)

# Get a list of the top 10 most frequently used words from the article
from heapq import nlargest
#print(nlargest(10, freq, key=freq.get))

# Creating a dictionary where the keys are the sentences and the values 
# are the significance scores
from collections import defaultdict
ranking = defaultdict(int)

for i, sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]

#print(ranking)

# Getting the top 4 highest ranked sentences
sents_idx = nlargest(4, ranking, key=ranking.get)
#print(sents_idx)

#print([sents[j] for j in sorted(sents_idx)])

# Adding all of this logic to a function
def summarize(text, n):
    sents = sent_tokenize(text)

    assert n <= len(sents)
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words("english") + list(punctuation))

    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)

    ranking = defaultdict(int)

    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]

    
    sents_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sents_idx)]

print(summarize(text, 3))