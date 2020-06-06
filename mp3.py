# Starter code for CS 165B HW3
import sys, os, os.path
import json
import numpy
import sklearn
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import pandas
import torch
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[string]
        training_label: List[string]
        testing_data: List[string]

    Output:
        testing_prediction: List[string]
    Example output:
    return ['NickLouth']*len(testing_data)
    """
    training_data = [x.lower() for x in training_data]
    training_data = [word_tokenize(x) for x in training_data]
    #print(train_data[0])
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(training_data):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word,tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        training_data[index] = str(Final_words)
    
    #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(training_data,training_labels,test_size=0.3)
    Train_X = training_data
    Train_Y = training_labels
    Test_X = testing_data

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    #Test_Y = Encoder.fit_transform(Test_Y)

    #Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(training_data)
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    #print(Tfidf_vect.vocabulary_)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    #print(Test_Y)
    #print(list(Encoder.inverse_transform(Test_Y)))
    #print("SVM Accuracy Score -> ",accuracy_score(Test_Y,predictions_SVM)*100)
    #print(Tfidf_vect.vocabulary_[0])
    #print(list(Encoder.inverse_transform(predictions_SVM)))
    return(list(Encoder.inverse_transform(predictions_SVM)))