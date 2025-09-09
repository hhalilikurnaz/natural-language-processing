#bir kelimenin farklı anlamları arasından doğru olanı bağlama göre seçme işlemidir.

import nltk
from nltk.wsd import lesk #kelimenin hangi anlamının oldugunu bulmak için çevresindeki diğer kelimelere bakar.

#gerekli nlkt paketleri 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

sentence1="The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities."
w1="bank"

sense1=lesk(nltk.word_tokenize(sentence1),w1)
print(f"Cumle : {sentence1}")
print(f"Word : {w1}")
print(f"Sense : {sense1.definition()}")


sentence2="The river bank was wet so ı can not sit."
w2="bank"

sense2=lesk(nltk.word_tokenize(sentence2),w2)
print(f"Cumle : {sentence2}")
print(f"Word : {w2}")
print(f"Sense : {sense2.definition()}")

#burada bank kelimesi 2 farklı anlamda kullanılmıştır. biri finans diğeri ise oturdumuz bank anlamında .

