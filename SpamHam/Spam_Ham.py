
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import time

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')
plt.style.use('seaborn-whitegrid')

import plotly.graph_objs as go
from plotly.offline import plot

import plotly.figure_factory as ff
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


data = pd.read_csv('Data/SMSSpamCollection',sep='\t',header=None,)
data.columns = ['Label', 'Message']
data.head()


# In[3]:


stop = set(stopwords.words('english'))
all_text = ' '

for sentence in data['Message']:
    
    sentence = str(sentence)
    words = sentence.split() 
    
    for i in range(len(words)): 
        words[i] = words[i].lower() 
          
    for word in words: 
        all_text = all_text + word + ' '

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop, 
                min_font_size = 10).generate(all_text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# In[4]:


data.describe()


# In[5]:


data.groupby('Label').describe()


# In[6]:


data['Message_Length'] = data['Message'].apply(len)


# In[7]:


hist_data = [np.array(data.loc[data['Label']=='ham']['Message_Length']),
            np.array(data.loc[data['Label']=='spam']['Message_Length'])]
group_labels = ['Message Length Ham','Message Length Spam']

fig = ff.create_distplot(hist_data, group_labels, curve_type='normal', show_hist=True, show_rug=False, show_curve=False)
plot(fig, filename='Histogram-MessageLength', auto_open=False)


# In[8]:


ps = PorterStemmer()
lemm = WordNetLemmatizer()
messages = list()
ts = time.time()
for message in data['Message']:
    clean = re.sub('[^a-zA-z]', ' ', message)
    clean = clean.lower()
    clean = clean.split()
    
    clean = [lemm.lemmatize(ps.stem(word), pos="v") for word in clean if not word in stopwords.words('english')]
    clean = ' '.join(clean)
    messages.append(clean)

print(time.time()-ts)


# In[9]:


cv = CountVectorizer()

X = cv.fit_transform(messages).toarray()
y = pd.get_dummies(data['Label'])
y = y.iloc[:,1].values
X.shape, y.shape


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


# In[11]:


model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(classification_report(y_test,pred))


# In[12]:


model = MultinomialNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(classification_report(y_test,pred))


# In[13]:


model = RandomForestClassifier(n_estimators=100, max_depth=2)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(classification_report(y_test,pred))

