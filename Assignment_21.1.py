
# coding: utf-8

# In[32]:


from bs4 import BeautifulSoup
import urllib.request
import nltk
from sklearn.feature_extraction.text import CountVectorizer
response = urllib.request.urlopen('http://php.net/')
html = response.read()
soup = BeautifulSoup(html,"html5lib")
htmltext=soup.getText()
#print(htmltext)
#htmltext = soup.prettify()
Y=[htmltext]
def get_top_n_words(corpus, n=None):
 vec = CountVectorizer().fit(Y)
 bag_of_words = vec.transform(Y)
 sum_words = bag_of_words.sum(axis=0)
 words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
 words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
 return words_freq[:n]

#frequency of top 30 words
common_words = get_top_n_words(Y,30)
for word, freq in common_words:
    print(word, freq)

