#Bag Of Words (BoW) metinlerdeki kelimeleri sayısal verilere dönüştürür ve metin analizi sağlar
#Nasıl çalışır metin içindeki kelimeleri sayısal gruplara ayırır frekans olarak atar mesela bir kelime 3 kere geçmişse frekansı 3 olur 

#İşleyişi : ilk amacımız kelime kümesi oluşturma 
#kelime frekansı hesaplama > Sonra vektör temsili 

#count vectorizer içeri aktar 
from sklearn.feature_extraction.text import CountVectorizer





#veri seti oluştur 
documents=["kedi bahçede",
           "kedi evde"]


#vectorizer tanımla 
vectorizer=CountVectorizer()


#metni sayısal vektörlere çevir 
X=vectorizer.fit_transform(documents) #önce fit et uygula ve sonra dönüştür demek

#kelime kümesi oluşturma 
feature_names=vectorizer.get_feature_names_out()
print(f"kelime kümesi : {feature_names}")


#vektör temsili 
vector_temsili=X.toarray()
print(f"vectör_temsili : {vector_temsili}")