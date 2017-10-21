import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pyLDAvis.gensim
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

with open("rt-polarity.txt")as fo:
    for rec in fo:
        doc_set.append(rec)

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatize = WordNetLemmatizer()


def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    three = " ".join(lemmatize.lemmatize(i) for i in two.split())
    return three


def pre_new(doc):
    one = cleaning(doc).split()
    two = dictionary.doc2bow(one)
    return two


print(ldamodel.print_topics(num_topics=2, num_words=4))
print(len(doc_set))

for i in ldamodel.print_topics():
    for j in i:
        print (j)

ldamodel.save('topic.model')
from gensim.models import LdaModel
loading = LdaModel.load('topic.model')
print(loading.print_topics(num_topics=2, num_words=4))

belong = loading[corpus[0]]
new = pd.DataFrame(belong,columns=['id','prob']).sort_values('prob',ascending=False)
new['topic'] = new['id'].apply(loading.print_topics)

#pyLDAvis.enable_notebook()
text_list = [k.split() for k in doc_set]
print(len(text_list))
print(text_list[0])

dictionary = corpora.Dictionary(text_list)
dictionary.save('dictionary.dict')
print(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)

print(len(doc_term_matrix))
print(doc_term_matrix[0])

d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.LdaModel.load('topic.model')

#data = pyLDAvis.gensim.prepare(lda, c, d)
