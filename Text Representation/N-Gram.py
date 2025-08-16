#dil modelinde kullanılan kelime veya karakter dizisinin uzunlugunu belirten bir terimdir
#N-Gram modelleri metinleri n kelimelik veya ne karakterlik kısımlara bölerek analiz eder 
'''
Metin modelleme için kullanılabilir 
Metin sınıflandırma için kullanılabilir 
Metin üretimi için Kullanılabilir 
Metin benzerliği için kullanılabilir 
'''

#import library
from sklearn.feature_extraction.text import CountVectorizer



#örnek metin

documents=["Bu çalışma Ngram çalışmasıdır. ",
"Bu çalışma doğal dil işleme çalışmasıdır"]

#unigram,bigram,treegram 3 farklı n değerine sahip  gram modeli yazıcaz 
vectorizer_unigram=CountVectorizer(ngram_range=(1,1))
vectorizer_bigram=CountVectorizer(ngram_range=(2,2))
vectorizer_treegram=CountVectorizer(ngram_range=(3,3))

#unigram
X_unigram=vectorizer_unigram.fit_transform(documents)
unigram_features=vectorizer_unigram.get_feature_names_out()

X_bigram=vectorizer_bigram.fit_transform(documents)
bigram_features=vectorizer_bigram.get_feature_names_out()

X_treegram=vectorizer_treegram.fit_transform(documents)
treegram_features=vectorizer_treegram.get_feature_names_out()

vektör_temsili_unigram=X_unigram.toarray()
print(f"\nunigram features : {unigram_features}")
print(f"\ unigram vektör temsili  : {vektör_temsili_unigram}")

vektör_temsili_bigram=X_bigram.toarray()
print(f"\nbigram features : {bigram_features}")
print(f"\nbigram  Vektör temsili  : {vektör_temsili_bigram}")

vektör_temsili_treegram=X_treegram.toarray()
print(f"\ntreegram features : {treegram_features}")
print(f"\ntreegram  vektör temsili  : {vektör_temsili_treegram}")



#sonuçların incelenmesi


