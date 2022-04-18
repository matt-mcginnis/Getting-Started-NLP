from turtle import title
import requests
from bs4 import BeautifulSoup


# Building a Corpus of Tech Articles
def getAllDoxyDonkeyPosts(url, links):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == 'Older Posts':
                #print(title, url)
                links.append(url)
                getAllDoxyDonkeyPosts(url, links)
        except:
            title = ''
    return


blogUrl = "http://doxydonkey.blogspot.in"
links = []
getAllDoxyDonkeyPosts(blogUrl, links)


# Understanding the Clustering Workflow
def getDoxyDonkeyText(testUrl):
    response = requests.get(testUrl).content
    soup = BeautifulSoup(response, "lxml")
    mydivs = soup.findAll("div", {"class":'post-body'})

    posts = []
    for div in mydivs:
        posts += map(lambda p:p.text, div.findAll("li"))

    return posts


doxyDonkeyPosts = []
for link in links:
    doxyDonkeyPosts += getDoxyDonkeyText(link)

#print(doxyDonkeyPosts[0:5])


# Identifying Themes using K-Means Clustering
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")

X = vectorizer.fit_transform(doxyDonkeyPosts)

#print(X[0])

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)

km.fit(X)

import numpy as np
np.unique(km.labels_, return_counts = True)

# Now that clusters have been created, we will identify some important keywords from each cluster.
text = {}
for i, cluster in enumerate(km.labels_):
    oneDocument = doxyDonkeyPosts[i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument

# Finding most frequent words in the clusters
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk

_stopwords = set(stopwords.words('english') + list(punctuation) + ["million", "billion", "year", "millions", "billions", "y/y", "'s", "''"])

keywords = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster] = freq

unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster]) - keys_other_clusters
    unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)

print(unique_keys)


# Classifying new article based on trained model.
article = "Alibaba Group Holding Ltd.’s six-year-old excursion into mobile operating systems is faltering in China, casting doubt over software that bears billionaire-founder Jack Ma’s name and was once touted as key to countering Tencent Holdings Ltd. China’s largest e-commerce company debuted YunOS in 2011, a system that underpins search, shopping and browsing that its executives last year said could attain as much as 25 percent domestic market share by the end of 2016 -- surpassing Apple Inc.’s iOS. Six years on, YunOS’ slice of China software installations stands at just 2.2 percent while its share of 2016 shipments was 10 percent, researchers Canalys and Counterpoint estimate, respectively. Alibaba disputes those numbers. Alibaba managers have grown increasingly unhappy with its sluggish adoption and have begun an internal debate around the software’s future, a person familiar with the matter said. No conclusions have been reached, the person said, asking not to be named discussing a confidential matter. Yet the talks reflect the inability of a once-vaunted initiative to forestall Tencent’s dominance in the mobile arena, secured through the utility of WeChat -- a universal app that melds messaging, payments, media, shopping and on-demand services."

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X, km.labels_)

test = vectorizer.transform([article])
#print(test[0])

print(classifier.predict(test))