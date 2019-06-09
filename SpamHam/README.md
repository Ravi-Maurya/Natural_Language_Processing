# Spam Ham Classification

Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
## Data Source
<b>SPAM HAM Data Set</b>
> <b><a href="https://archive.ics.uci.edu/ml/datasets/sms+spam+collection">UCI Collecttion</a></b>

## Libraries

> <b>Numpy</b>: 1.16.0<br>
> <b>Pandas</b>: 0.23.4<br>
> <b>Matplotlib</b>: 3.0.2<br>
> <b>Scikit Learn</b>: 0.20.2<br>
> <b>Wordcloud</b>: 1.5.0<br>
> <b>Plotly</b>: 3.10.1<br>

# Prediction Models
Model used are from simple scikit-learn library and the procesiing is done from scratch as it does not use any pretrained model.
## Pre Processing
<ul>
  <li>Lower Casing</li>
  <li>Removing unwanted things by RE</li>
  <li>Lemmatization for root word cause</li>
  <li>Vectorizing i.e. text to number transformation</li>
</ul>

## Naive Bayes
In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

Naive Bayes has been studied extensively since the 1960s. It was introduced (though not under that name) into the text retrieval community in the early 1960s and remains a popular (baseline) method for text categorization, the problem of judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as the features. With appropriate pre-processing, it is competitive in this domain with more advanced methods including support vector machines. It also finds application in automatic medical diagnosis.

Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. 

<h3>Assumption:</h3>

The fundamental Naive Bayes assumption is that each feature makes an:
<ul>
<li>independent</li>
<li>equal</li>
</ul>
contribution to the outcome.

With relation to our dataset, this concept can be understood as:

<li>We assume that no pair of features are dependent. For example, the temperature being ‘Hot’ has nothing to do with the humidity or the outlook being ‘Rainy’ has no effect on the winds. Hence, the features are assumed to be independent.</li>
<li>Secondly, each feature is given the same weight(or importance). For example, knowing only temperature and humidity alone can’t predict the outcome accuratey. None of the attributes is irrelevant and assumed to be contributing equally to the outcome.</li>
<br>
