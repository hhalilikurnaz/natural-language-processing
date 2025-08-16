#stemming(kök bulma )
#stemming kelimelerin kök formunu bulmaki için kelimenin sonundakş eklerin(suffix) çıkarılması işlemidir
#kelimenin tamamen anlamını dogru bir şeklilde bulmayı amaçlamaz daha ziyade kelimenin en basit formunu bulmaya odaklanır 

#lemmatization gövdeleme 
'''
kelimeleri sözlükteki temel fromlarına dönğüştürme işlemidir 
kelimenin anlamını ve dilbilgisel yapısını dikkate alarak doğru bir kök bulmaya calılır 
bu nedenle lemmatiation sonrası elde edilen kelime dilgibisel olarak anlamlı ve sözlğkte yer alan bir kelime olur '''

import nltk 
nltk.download("wordnet") #wordnet : lemmatization islemi için gerekli veri tabanı 

from nltk.stem import PorterStemmer # stemming için fonksiyon

#porter stemmer nesnesi oluştur
stemmer=PorterStemmer()
words=["running","runner","ran","rans","better","go","went"]
#kelimelerin stem'lerini buluyoruz,bunu yaparken de porter stemmerşn stem() fonksiyonunu kullanıyoruz 
stems=[stemmer.stem(w) for w in words ]
print(stems)



#\\\\\\\\\###
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words=["running","runner","ran","rans","better","go","went"]
lemmas=[lemmatizer.lemmatize(w,pos="v") for w in words]
print(f"Lemas : {lemmas}")