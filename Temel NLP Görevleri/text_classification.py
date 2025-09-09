#metinleri otomatik olarak belirli kategorilere ayırma

#nerelerde kullanılıyor : e posta filtreleme, müşteri geri bildirim analizi, haber sınıflandırma

#metni doğruan input olarak veremiyoruz .bunun için feature extraction yapmamız gerekiyor.(sayısallaştırılmış vektörler)
#metineri sınıflandırıken akış semalası 
'''
raw text -> tokenization -> text_cleaning -> POS tagging -> stopwords -> lemmetization -> cleaned text -> ML model 
'''

''' spam veri seti var -> spam ve ham -> binary classification with decision tree'''

#import libraries 
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score

import nltk
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer








#load dataset 


data = pd.read_csv("/Users/halilibrahimkurnaz/Desktop/NLP/Temel NLP Görevleri/data.csv", encoding='latin-1')
print(data.head(5))

data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True) #gereksiz featureları kaldırdık)
data.columns = ['label','text'] #kolon isimlerini değiştirdik



#EDA (Keşifsel veri analizi -> Exploratory Data Analysis) : missing value ??
print(data.isnull().sum())



#text cleaning and preprocessing:ozel karakterler ,kucuk harfler yapmak,tokenization,stopwords,lemmetization

nltk.download('stopwords')
nltk.download('wordnet') #lemma bulmak için gerekli veri seti 
nltk.download('omw-1.4') #wordnete ait farklı dillerin kelime anlamlırnı içeren bir veri seti 

text=list(data.text)
lemmatizer=WordNetLemmatizer()

corpus=[]


for i in range(len(text)):
    r=re.sub('[^a-zA-Z]',' ',text[i]) #özel karakterleri kaldırdık
    r=r.lower()
    
    r=re.sub(r'\d+',' ',r) #sayısal verileri kaldırdık
    r=r.split() #tokenization

    r= [word for word in r if word not in stopwords.words("english")] # stopwords'lerden kurtulduk 
    r=[lemmatizer.lemmatize(word) for word in r ] #kelimelerin kök halini aldık

    r=" ".join(r)
    corpus.append(r)

    

data['text2']=corpus
print(data.head(5))



#model training and evaluation(değerlendirilmesi)

X=data['text2']
y=data['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#feature extraction (metinleri sayısal verilere dönüştürme) : bag of words 

cv=CountVectorizer()
X_train_cv=cv.fit_transform(X_train)


#classifier training : model training and evaluation

dt=DecisionTreeClassifier()
dt.fit(X_train_cv,y_train) # egitim 

x_test_cv=cv.transform(X_test) #test verisini dönüştürdük

y_pred=dt.predict(x_test_cv) #tahmin

c_matrix=confusion_matrix(y_test,y_pred)
print(c_matrix)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy : {accuracy}")