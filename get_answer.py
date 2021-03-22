#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:20:06 2021

@author: harshit
"""

import pickle


with open('labelencoder', 'rb') as f:
    labelencoder = pickle.load(f) 
    
with open('onehotencoder', 'rb') as f:
    onehotencoder = pickle.load(f) 
    
with open('tokenizer', 'rb') as f:
    tokenizer = pickle.load(f) 
    
    

    
    
import sys
import pandas as pd

text = sys.argv[1:]

text = ' '.join(text)
#print(text)

#text = "are"

df = pd.DataFrame(text.split(), columns=['Words'])

data = df.to_numpy()

df["Words"] = df["Words"].apply(str)

df = df['Words'].fillna('').tolist()


df = tokenizer.texts_to_sequences(df)


from keras.preprocessing import sequence

df = sequence.pad_sequences(df, maxlen=1)

from tensorflow import keras
model = keras.models.load_model('rnn.hdf5')


answer = model.predict(df)

output = labelencoder.inverse_transform(onehotencoder.inverse_transform(answer))

print()
print("*******")

print("This model has accuracy of 80%")

print("Answer for Text", text, "is : ")

for i in range(len(data)):
    print(data[i][0], ":", output[i])