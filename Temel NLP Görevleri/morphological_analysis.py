#kelimelerin yapısını inceleyerek dil bilgisel özellikleri belirler.
'''Kullanım alanları:
dil öğrenme araçları
nlp
otomatik çeviri'''

import spacy

nlp=spacy.load("en_core_web_sm") #spacy kutuphnaesi ingilice dil modeli

#incelenecek olan kelime ya da kelimeler:

word="My name is Halil .I am software engineer.I am handsome and ı try to improve myself on natrual language processing."

#kelimeyi nlp işleminden geçiriyoruz 


doc=nlp(word)

for token in doc:
    print(f"Text : {token.text}")
    print(f"Lemma : {token.lemma_}") #kelimenin kökü
    print(f"POS : {token.pos_}") #kelimenin dil bilgisel özelliği   
    print(f"Tag : {token.tag_}") #daha detaylı dil bilgisel özellik
    print(f"Dep : {token.dep_}") #kelimenin cümledeki görevi(yüklem,özne vs)
    print(f"Shape : {token.shape_}")#kelimenin yapısı (büyük küçük harf, rakam vs)
    print(f"Is alpha : {token.is_alpha}") #sadece harflerden mi oluşuyor    
    print(f"Is stop : {token.is_stop}") #önemsiz kelime mi (bağlaç, edat vs)
    print(f"Morphology : {token.morph}") #kelimenin morfolojik özellikleri (tekil, çoğul, zaman vs)
    print(f"Is plural : {'Number=Plur' in token.morph}") #çoğul mu (True/False)
    
    print()
    print("----")
    print()