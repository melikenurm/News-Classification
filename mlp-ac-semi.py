# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 03:39:13 2019

@author: Melike Nur Mermer
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:19:44 2019

@author: Melike Nur Mermer
"""

import numpy
from sklearn.neural_network import MLPClassifier
from os import listdir
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            ff=open(doc, 'r', encoding='utf-8').read()
            yield TaggedDocument(words=ff.split(), tags=[self.labels_list[idx]])

files=["dunya","ekonomi","kultursanat","magazin","saglik","siyaset","spor","teknoloji","yasam"]
#files=["dunya","ekonomi"]
docLabels = []
data = []
labels= []

for i in range(len(files)):
    docLabels = [f for f in listdir("egitim/"+files[i]) if f.endswith('.txt')]

    for doc in docLabels:
        #ff=open("egitim/" + files[i] +"/" + doc, 'r', encoding='utf-8')
        data.append("egitim/" + files[i] +"/" + doc)
        labels.append(files[i])
        
it = LabeledLineSentence(data, labels)

model = Doc2Vec(size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025) # sabit lr
model.build_vocab(it)

for epoch in range(4):
    model.train(it, total_examples=model.corpus_count, epochs=epoch+1)
    model.alpha -= 0.002 # 1,3,6 epochlarda decrease lr 
    model.min_alpha = model.alpha

train_data = []
train_labels = []

for i in range(len(files)):
    docLabels = [f for f in listdir("egitim/"+files[i]) if f.endswith('.txt')]

    for doc in docLabels:
        ff=open("egitim/" + files[i] +"/" + doc, 'r' , encoding='utf-8')
        train_data.append(model.infer_vector(ff.read().split())) #metinlerin vektörleri alınıyor
        train_labels.append(files[i])
        ff.close()

test_data = []   
test_labels = []
test_pred = []

for i in range(len(files)):
    docLabels = [f for f in listdir("test/"+files[i]) if f.endswith('.txt')]

    for doc in docLabels:
        ff=open("test/" + files[i] +"/" + doc, 'r' , encoding='utf-8')
        inferred_vector = model.infer_vector(ff.read().split())
        test_data.append(inferred_vector)
        #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        #test_pred.append(sims[1])
        test_labels.append(files[i])
        ff.close()
        
#baseline classifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=0, warm_start=True) #tüm yöntemler için aynı başlangıç
classifier.fit(train_data, train_labels)
print("Baseline is..: "+str(classifier.score(test_data, test_labels)))

#semi-supervised learning - self training algorithm
labeled_data=[]
labeled_labels=[]
unlabeled_data=[]
unlabeled_labels=[] #for curriculum
unlabeled_pred=[]

for i in range(len(train_labels)):
    if i%2==0:
        labeled_data.append(train_data[i])
        labeled_labels.append(train_labels[i])
    else:
        unlabeled_data.append(train_data[i])
        unlabeled_labels.append(train_labels[i]) #for curriculum

semi_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=0, warm_start=True) # warm_start -> kaldığı yerden devam
semi_classifier.fit(labeled_data, labeled_labels)
print("Semi-supervised first half is..: "+str(semi_classifier.score(test_data, test_labels)))

all_data=[]
new_labels=[]

unlabeled_pred=semi_classifier.predict(unlabeled_data)

for i in range(len(unlabeled_pred)):
    for j in range(len(files)):
        if(unlabeled_pred[i]==files[j]):
            new_labels.append(files[j])
all_data=labeled_data+unlabeled_data
all_labels=labeled_labels+new_labels
semi_classifier.fit(all_data, all_labels)
print("Semi-supervised finally is..: "+str(semi_classifier.score(test_data, test_labels)))

#anti-curriculum
next_train_data=[]
next_train_labels=[]
new_unlabeled_data=[]
new_unlabeled_labels=[]

curr_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=0, warm_start=True)
curr_classifier.fit(labeled_data, labeled_labels)
print("Anti-Curriculum first half is..: "+str(curr_classifier.score(test_data, test_labels)))
proba=curr_classifier.predict_proba(unlabeled_data)
proba=proba.max(axis=1)
sorted_proba=numpy.argsort(proba) #zordan kolaya indexler

for i in range(round(len(proba)/2)):
    next_train_data.append(unlabeled_data[sorted_proba[i]])
    next_train_labels.append(unlabeled_labels[sorted_proba[i]])
    new_unlabeled_data.append(unlabeled_data[sorted_proba[round(len(proba)/2)-1+i]])
    new_unlabeled_labels.append(unlabeled_labels[sorted_proba[round(len(proba)/2)-1+i]])

curr_data=labeled_data+next_train_data
curr_labels=labeled_labels+next_train_labels
curr_classifier.fit(curr_data, curr_labels) #ikinci yarıdaki zorlar eklenmiş
print("Anti-Curriculum with hard quarter is..: "+str(curr_classifier.score(test_data, test_labels)))
  
next_curr_data=curr_data+new_unlabeled_data
next_curr_labels=curr_labels+new_unlabeled_labels
curr_classifier.fit(next_curr_data, next_curr_labels) #ikinci yarı tamamen eklenmiş
print("Anti-Curriculum finally is..: "+str(curr_classifier.score(test_data, test_labels)))
