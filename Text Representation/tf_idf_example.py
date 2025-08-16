#import libraries 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
import nltk 
from nltk.corpus import stopwords


#veri seti yükle 
df = pd.read_csv("Text Representation/spam.csv", encoding="latin1")
print(df.head(4))

#veri temizleme bloğu 
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)       # URL kaldır
    text = re.sub(r"\d+", "", text)                           # Sayı kaldır
    text = re.sub(r"[^\w\s]", "", text)                       # Noktalama kaldır
    words = text.split()                                      # Kelimelere ayır
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)                                    # Tekrar birleştir

df["text"] = df["v2"].apply(clean_text)




# 4. TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])


#tfidf
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(df["text"])
#kelime kümesini incele 
feature_names=vectorizer.get_feature_names_out()

tfidf_score=X.mean(axis=0).A1



#tfidf skorlarını içeren df oluştur 

df_tfidf=pd.DataFrame({"word":feature_names,"tfidf_score":tfidf_score})
print(df_tfidf)


#skorları sırala ve sonucları incele 
df_tfidf_sorted=df_tfidf.sort_values(by="tfidf_score",ascending=False)
print(df_tfidf_sorted.head(10))