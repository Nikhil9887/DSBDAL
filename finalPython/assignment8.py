# %%
import math
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('average-perceptron-tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# %%
with open('doc1.txt', 'r') as f:
    doc1 = f.read()

doc1 = re.sub('[\W_]+', ' ', doc1)

with open('doc2.txt', 'r') as f:
    doc2 = f.read()

doc2 = re.sub('[\W_]+', ' ', doc2)

# %%
doc1

# %%
doc2

# %%
word_tokens_1 = nltk.word_tokenize(doc1)

# %%
word_tokens_1

# %%
word_tokens_2 = nltk.word_tokenize(doc2)

# %%
word_tokens_2

# %%
all_tokens = word_tokens_1 + word_tokens_2
all_tokens

# %%
sent_tokenize = nltk.sent_tokenize(doc1)
sent_tokenize

# %%
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words

# %%
all_tokens = [token for token in all_tokens if token not in stop_words]
all_tokens

# %%
tags = nltk.pos_tag(all_tokens)
print(tags)

# %%
# lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in all_tokens]
lemmatized_tokens

# %%
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in all_tokens]
stemmed_tokens

# %%
word_tokens_1


def tf(tokens):
    term_freq = {}
    for token in tokens:
        if token not in term_freq:
            term_freq[token] = tokens.count(token) / len(tokens)

    return term_freq


doc1_tf = tf(word_tokens_1)
print(doc1_tf)

# %%


def idf_calculate():
    N = 2
    alltokens = word_tokens_1 + word_tokens_2
    idf = {}
    for token in alltokens:
        f = 0
        if (token in word_tokens_1):
            f += 1
        if (token in word_tokens_2):
            f += 1
        # idf[token] = math.log(N/f)
        if (f == 0):
            idf[token] = math.log(N/(f+1))
        else:
            idf[token] = math.log(N/f)

    return idf


idf = idf_calculate()
print(idf)

# %%


def tf_idf(tokens, tf, idf):
    ans = []
    for token in tokens:
        ans.append(tf[token] * idf[token])

    return ans


ans = tf_idf(word_tokens_1, doc1_tf, idf)
print(ans)
