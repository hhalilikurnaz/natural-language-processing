#metin parçası etiketleme part of speech (POS) tagging
# bir metindeki her kelimenin dilbilgisel kategorisini belirleyerek cümlelerin dil yapısını analiz etmeyi sağlar.


import spacy

nlp=spacy.load("en_core_web_sm")

sentence1="What is the weather like today or tomorrow?"
sentence2="Halil is very handsome boy and genius clever."

doc1=nlp(sentence1)
doc2=nlp(sentence2)

for token in doc2:
    print(token.text,token.pos_)
    print("-----")

for token in doc1:
    print(token.text,token.pos_)