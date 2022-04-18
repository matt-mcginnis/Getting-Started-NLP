import nltk

text = "Mary had a little lamb. Her fleece was white as snow."
from nltk.tokenize import word_tokenize, sent_tokenize

# Tokenizing sentences and words
print("Tokenizing Sentences and Words")

sents = sent_tokenize(text)
print(sents)

words = [word_tokenize(sent) for sent in sents]
print(words)

# Removing Stopwords
print("\nRemoving Stopwords")

from nltk.corpus import stopwords
nltk.download("stopwords")
from string import punctuation
customStopWords = set(stopwords.words("english") + list(punctuation))

wordsMinusStopwords = [word for word in word_tokenize(text) if word not in customStopWords]
print(wordsMinusStopwords)

# Identifying Bigrams
print("\nIdentifying Bigrams")

from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsMinusStopwords)
print(sorted(finder.ngram_fd.items()))

# Stemming and Parts of Speech Tagging
print("\nStemming and Parts of Speech Tagging")

text2 = "Mary closed on closing night when she was in the mood to close."
from nltk.stem.lancaster import LancasterStemmer
st=LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)

nltk.download("averaged_perceptron_tagger")
print(nltk.pos_tag(word_tokenize(text2)))

# Word Sense Disambiguation
print("\nWord Sense Disambiguation")

from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download("omw-1.4")
for ss in wn.synsets("bass"):
    print(ss, ss.definition())

print("\nDetermining the meaning of words in a phrase")

from nltk.wsd import lesk
sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass."), "bass")
print(sense1, sense1.definition())

sense2 = lesk(word_tokenize("This sea bass was really hard to catch."), "bass")
print(sense2, sense2.definition())