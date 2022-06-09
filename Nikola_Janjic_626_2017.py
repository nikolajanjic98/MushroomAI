import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import tensorflow as tf 
from tensorflow.keras.models import Sequential




#ucitavanje
data = pd.read_csv(r"C:\Users\Win10\Desktop\Nikola_Janjic_VI\mushroom\mushroom_dataset_.csv")


#da vidimo koje vrednosti moye imati klasa
data["Class"].unique()
print("\n Jedinstvene vrednosti atributa: \n")
print(data["Class"].unique())


#provera da li odredjeni atribut ima jedinstvenu vrednost
print("\n da li odredjeni atribut ima jedinstvenu vrednost: \n")
print(data.nunique())



#ovde krece crtanje atributa


plt.figure(figsize=(10,5))
sns.countplot(x=data['Cap-shape'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Cap shape')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Cap-surface'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Cap surface')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x=data[' Cap-color'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Cap color')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(x=data[' Bruises'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Bruises')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Odor'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('odor')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Gill-attachment'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('gill-attachment')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Gill-spacing'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('gill-spacing')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Gill-size'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('gill-size')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Gill-color'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('gill-color')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-shape'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-shape')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-root'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-root')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-surface-above-ring'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-surface-above-ring ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-surface-below-ring'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-surface-above-ring ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-color-above-ring'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-color-above-ring ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Stalk-color-below-ring'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Stalk-color-below-ring ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Veil-type'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Veil-type ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Veil-color'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Veil-color ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Ring-number'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Ring-number ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Ring-type'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Ring-type ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Spore-print-color'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Spore-print-color ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Population'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Population ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data[' Habitat'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Habitat ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=data['Class'],hue=data['Class'], alpha =.80, palette= ['yellow','orange'])
plt.title('Class ')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')
plt.show()

#ovde se zavrsava crtanje atributa


#izbacivanje atributa Veil-type jer ima jednistvenu vrednost
data.drop(" Veil-type",axis=1,inplace=True)
#vidi da li mora ovo inplace=True

#promena vrednosti atributa Class
#ako je vrednost p postavlja se na 0, ako je vrednost e onda na 1
data["Class"] = [0 if i == "p" else 1 for i in data["Class"]]

#pretvaranje podataka iz tekstualnih u numericke vrednosti

for column in data.drop(["Class"], axis=1).columns:
    value = 0
    step = 1/(len(data[column].unique())-1)
    for i in data[column].unique():
        data[column] = [value if letter == i else letter for letter in data[column]]
        value += step



#neuronska mreza

y = data["Class"].values                                                                    
x = data.drop(["Class"], axis=1).values                                                     
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=0.25)      


print("\n Sve epohe neuronske mreze: \n")

from sklearn.preprocessing import MinMaxScaler

x_train,x_val,y_train,y_val=train_test_split(x_train, y_train, test_size=0.25,random_state=0)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledx_train = scaler.fit_transform(x_train)
rescaledx_test = scaler.fit_transform(x_test)
rescaledx_val=scaler.fit_transform(x_val)

x_train_normal = rescaledx_train
x_val_normal = rescaledx_val


tf.random.set_seed(0)
model=tf.keras.models.Sequential(layers=[
tf.keras.layers.Dense(50, activation="relu"),
tf.keras.layers.Dense(100,activation='relu'),
tf.keras.layers.Dense(2,activation="softmax")
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.SGD(0.1),
metrics=["accuracy"])
proces = model.fit(x_train_normal,y_train,epochs=20,validation_data=
(x_val_normal,y_val))

print(model.summary())


#evaluacija

print ("\n")
print("Evaluacija")
print("Evaluacija na testnim podacima")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test accuracy:", results)
print("\n")
print("Generisanje predvidjanja")
predictions = model.predict(x_test[:3])
print("Predvidjanja:", predictions.shape)


#Random Forest Classifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)
print("Random Forest Classifier preciznost testa: {}%".format(round(rf.score(x_test,y_test)*100,2)))

#crtanje konfuzionih matrica

y_pred_rf = rf.predict(x_test)
y_true_rf = y_test
cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()

print("\n X TRAIN SHAPE")
print(x_train.shape)
print("\n")

print("\n X TEST SHAPE")
print(x_test.shape)
print("\n")











        
