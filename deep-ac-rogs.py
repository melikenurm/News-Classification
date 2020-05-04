# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:01:38 2019

@author: Melike Nur Mermer
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:45:48 2019

@author: Melike Nur Mermer
"""

import pandas as pd
import numpy as np
import pickle
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, clone_model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer

# For reproducibility
np.random.seed(1237)
 
# Source file directory
files=["dunya","ekonomi","kultursanat","magazin","saglik","siyaset","spor","teknoloji","yasam"]
 

train_posts = []
train_tags = []

for i in range(len(files)):
    docLabels = [f for f in listdir("egitim/"+files[i]) if f.endswith('.txt')]

    for doc in docLabels:
        ff=open("egitim/" + files[i] +"/" + doc, 'r' , encoding='utf-8')
        train_posts.append(ff.read()) #metinlerin vektörleri alınıyor
        train_tags.append(files[i])
        ff.close()

test_posts = []   
test_tags = []
test_pred = []

for i in range(len(files)):
    docLabels = [f for f in listdir("test/"+files[i]) if f.endswith('.txt')]

    for doc in docLabels:
        ff=open("test/" + files[i] +"/" + doc, 'r' , encoding='utf-8')
        test_posts.append(ff.read())
        #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        #test_pred.append(sims[1])
        test_tags.append(files[i])
        ff.close()

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(train_posts))

# 20 news groups
num_labels = 9
vocab_size = 15000 #total=59853
batch_size = 32
 
# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)
 
x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')
 
encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

anti_model = clone_model(model)
anti_model.set_weights(model.get_weights())
 
rogs_model = clone_model(model)
rogs_model.set_weights(model.get_weights())

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=2,
                    verbose=1,
                    validation_split=0.0,
                    shuffle=True)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
 
print('Test accuracy:', score[1])

#anti-curriculum
stages=3
labeled_data=[]
labeled_labels=[]

for i in range(len(y_train)):
    if i%stages==0:
        labeled_data.append(x_train[i])
        labeled_labels.append(y_train[i])
        
labeled_data=np.asarray(labeled_data)
labeled_labels=np.asarray(labeled_labels)

anti_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

anti_model.fit(labeled_data, labeled_labels,                     
               batch_size=batch_size,
               epochs=1,
               verbose=1,
               validation_split=0.0,
               shuffle=True)
print("Anti-Curriculum first stage is..: "+str(anti_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]))

for j in range(2, stages+1):
    curr_data=[]
    curr_labels=[]
    proba=anti_model.predict(x_train)
    proba=proba.max(axis=1)
    sorted_proba=np.argsort(proba) #zordan kolaya indexler

    for i in range(int(len(proba)/stages)*j):
        curr_data.append(x_train[sorted_proba[i]])
        curr_labels.append(y_train[sorted_proba[i]])

    curr_data=np.asarray(curr_data)
    curr_labels=np.asarray(curr_labels)
    
    anti_model.fit(curr_data, curr_labels,                     
               batch_size=batch_size,
               epochs=1,
               verbose=1,
               validation_split=0.0,
               shuffle=True)
    #anti_model.alpha -= 0.002
    print("Anti-Curriculum stages is..: "+str(anti_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]))

#rogs
labeled_data=[]
labeled_labels=[]

for i in range(len(y_train)):
    if i%stages==0:
        labeled_data.append(x_train[i])
        labeled_labels.append(y_train[i])
        
labeled_data=np.asarray(labeled_data)
labeled_labels=np.asarray(labeled_labels)

rogs_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

rogs_model.fit(labeled_data, labeled_labels,                     
               batch_size=batch_size,
               epochs=1,
               verbose=1,
               validation_split=0.0,
               shuffle=True)

print("ROGS first stage is..: "+str(rogs_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]))

for j in range(2, stages+1):
    curr_data=[]
    curr_labels=[]
    new_data=[]
    new_labels=[]
    
    for i in range(len(y_train)):
        if i%stages==(j-1):
            new_data.append(x_train[i])
            new_labels.append(y_train[i])

    new_data=np.asarray(new_data)
    new_labels=np.asarray(new_labels)
    
    labeled_data=np.concatenate((labeled_data,new_data))
    labeled_labels=np.concatenate((labeled_labels,new_labels))
    
    rogs_model.fit(labeled_data, labeled_labels,                     
               batch_size=batch_size,
               epochs=1,
               verbose=1,
               validation_split=0.0,
               shuffle=True)
    #anti_model.alpha -= 0.002
    print("ROGS stages is..: "+str(rogs_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)[1]))
