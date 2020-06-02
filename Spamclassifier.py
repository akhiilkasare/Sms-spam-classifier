# -*- coding: utf-8 -*-

# Importing the dataset

import pandas as pd

message = pd.read_csv('/home/akhil/Downloads/machine_learning/natural_language_processing/spam_classifier/SMSSpamCollection',
                      sep = '\t', names = ["label", "message"])

# Data Cleaning & Data Preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

corpus = []

for i in range (len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review  if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(message['label'])
y = y.iloc[:,1].values

# Train Test Splitting

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#Training our model on naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred = spam_detect_model.predict(x_test)


from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)











