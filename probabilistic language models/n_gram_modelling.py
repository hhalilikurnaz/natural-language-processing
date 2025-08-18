#ngram modelleri bir dizideki ardıık kelime veya kaarater gruplarının olasılıklarını tahmin eder 
#uni \ bi\tree-gram modelleri gibi

#avantajlar basit ve hızlı yerel bağlantıları iyi yakalar 
# dezavantajlar uzun bağımlılıkları yakalayamaz ve kelime sırasını dikkate almaz


#import libraries 
import nltk
from nltk.util import ngrams #n-gram omodeli oluşturmak için
from nltk.tokenize import word_tokenize # tokenization 

from collections import Counter # kelime sayımı için

#öernek veri seti 
corpus=[""
"Ilove apple pie",
"Ilove banana pie",
"Ilove NLP",
"He loves apple ",
"They love apple ",
"I love you and you love me ",
"I love fenerbahçe and fenerbahce makes me happy",
"I love you"]

"""problem tahmini yapalım : 
dil modeli yapmak istiyoruz 
amac 1 jkelimeden sonra gelecek kelimeyi tahmin etmek : metin türetmek/ oluşturmak
bunun için n-gram dil modelini  kullanacağız

ex: I ...(love) noktayı doldurdu love dan sonra ne gelecek ? olasılıksal olarak apple yazıcak.


"""



# verileri token haline getirme (tokenization)
tokens=[word_tokenize(sentence.lower()) for sentence in corpus]



#iikili ve üçlü kelime gruplarını oluşturma (bigrams ve trigrams)

bigrams=[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list,2)))  #2 kelimeli gruplar oluşturma
bigrams_freq=Counter(bigrams)




treegram=[]
for token_list in tokens:
    treegram.extend(list(ngrams(token_list,3)))  #3 kelimeli gruplar oluşturma
treegram_freq=Counter(treegram)



#model testing 


# I love bigramından sonra "you" or "apple " kelimelerinin gelme olasılıklarını hesaplama 

bigram=("i","love") #hedef bigram

#"i  love you "olma olasığı 
prob_you=treegram_freq[("i","love","you")]/bigrams_freq[bigram]

print(f"you kelimesinin olma olasığı : {prob_you}")


#" ı love apple" olma olasılığı 
prob_apple=treegram_freq[("i","love","apple")]/bigrams_freq[bigram]
print(f"apple kelimesinin olma olasığı : {prob_apple}")