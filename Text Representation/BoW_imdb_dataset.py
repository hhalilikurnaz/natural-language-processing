#import libraries 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import re 
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_word_english=set(stopwords.words("english"))



#import dataset 
df = pd.read_csv("Text Representation/dataset.csv")
print(df.head(5))

#metin verilerini alalım
documents=df["review"]
labels=df["sentiment"] #positive or negative 

#metin temizleme(veri temizleme )
def clean_text(text):
 
    #büyük küçük harf çevirme 
    text=text.lower()



    #rakam temizleme 
    text=re.sub(r"\d+", "",text)


    #özel karakterlerin kaldırılması 
    text=re.sub(r"[^\w\s]","",text)


    #kısa kelimelerin temizlenmesi (I,in)
    text=" ".join([word for word in text.split() if len(word) >2])



    #stop words çıkarımı 
    text_list=text.split()
    filtered_words=[word for word in text_list if word.lower() not in stop_word_english]





    return " ".join(filtered_words) #temizlenmiş texti return et

#metinleri temizle 
cleaned_doc=[clean_text(row) for row in documents]



#BoW

#vectorizer tanımla 
vectorizer=CountVectorizer()



#metinleri sayısal hala getir 
X=vectorizer.fit_transform(cleaned_doc[:75])



#kelime kümesi göster 
feature_names=vectorizer.get_feature_names_out()
print(f" feature names {feature_names}")


#vector temsili göster 
vector_temsili=X.toarray()
print(f"Vector Temsili : {vector_temsili}")


df_bow=pd.DataFrame(vector_temsili,columns=feature_names)
print(df_bow)

#kelime frekans göster 
word_count=X.sum(axis=0).A1
word_freq=dict(zip(feature_names,word_count))
print(f"word count : {word_count} \t word freq {word_freq}")


#ilk 5 kelime 
word_commen_5_words=Counter(word_freq).most_common(5)
print(f"most_common_5_words: {word_commen_5_words}" )

