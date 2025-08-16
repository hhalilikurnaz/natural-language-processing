'''
TF-IDF (Term Frequency-Inverse Document Frequency ) kelimelerin belgeler içinde ne kadar önemli oldugunu belirlemek için kullanılır 
Term Frequency (TF) Kelimelerin ne kadar sık geçtiğine bakar çok sık geçen çok önemlidir der 
Inverse Document Frequency Kelimenin tüm belgedeki yaygınlığını ölçer.Bir kelime çok belgede geçiyorsa o kelime çok fazla bilgi sağlamaz der 
'''

#import libraries 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
#create document 
document=["Kopekler cok tatlı hayvanlar",
          "Halil cok tatlı ve yakısıklı",
          "Kopekler ve kuşlar çok tatlı hayvanlar ",
          "inekler sut üretirler"]

#define vectorizer 
tfıdf_vectorizer=TfidfVectorizer()


#metinleri sayısal hale getir 
X=tfıdf_vectorizer.fit_transform(document)


#kelime kümesini incele 
feature_names=tfıdf_vectorizer.get_feature_names_out()
print(f"Feature names : {feature_names}")

#vektör temsilini incele 
vektör_temsili=X.toarray()
print(f"Vektör temsili {vektör_temsili}")

dt_tfidf=pd.DataFrame(vektör_temsili,columns=feature_names)
print(dt_tfidf)

#ortalama tf idf değerlerine bakalım 
tf_idf=dt_tfidf.mean(axis=0)
print(tf_idf)
