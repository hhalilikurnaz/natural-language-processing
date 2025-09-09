#bir metin içerisindeki kişi yer organizasyon gibi özel isimleri tanımlayarak anlamlı bilgiler çıkarmaya yarar

'''
bilgi çıkarma 
otomatik özetleme 
müşteri ilişkileri'nde kullanırız'''

'''problem:
varlık ismi tanima : metin(text) -> metin içerisinde bulunan varlık ısımlerini tanımla '''



#impoer libraries 
import pandas as pd 
import spacy




#spacy modeli ile varlik ismi tanımla 
nlp=spacy.load("en_core_web_sm") #spacy kutuphnaesi ingilice dil modeli 

content="Alice works at Amazon and lives in London.She visited British Museum in 2020."

doc=nlp(content) #contenti nlp dil modeline verdil -> bu işlem metindeki varlıkları (entitiees) tanımlamak için yapılır

for ent in doc.ents:
    #ent.text -> varlık ismi (Alice,Amazon...)
    #ent.start_char -> varlık isminin metin içerisindeki başlangıç karakteri
    #ent.end_char -> varlık isminin metin içerisindeki bitiş karakteri
    #ent.label_ -> varlık isminin türü (PERSON,ORG,LOC)

    print(ent.text,ent.label_)
    
entities=[(ent.text,ent.label_,ent.lemma_) for ent in doc.ents]

#varlık listesini df çeviriyoruz
df=pd.DataFrame(entities,columns=["text","type","Lemma"])
print(df)