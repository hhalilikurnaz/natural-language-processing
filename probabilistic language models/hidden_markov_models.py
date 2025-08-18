#bir dizi kelimelerin arkasında gizli bir durum dizisinin olduğu varsayımına dayanır 

"""Kullanım alanları:
konusma tanıma 
dil modelleme 
parça etiketleme 
Avantajları:
Bağlam modelleme 
Verşmli alogitmalar
Dezavantajları:
Sınırlı bağlam
Eğitim zorluğu
"""


#Part of Speech (POS) tagging yapıcaz (kelimeleri uygun sözcük türünü bulma calısması) HMM

"""
I (Zamir ) am a teacher (Noun)
"""

#import libraries
import nltk
from nltk.tag import hmm



#training data 
train_data=[
    (['I', 'am', 'a', 'teacher'], ['PRON', 'VERB', 'DET', 'NOUN']),
    (['He', 'is', 'a', 'doctor'], ['PRON', 'VERB', 'DET', 'NOUN']),
    (['They', 'are', 'students'], ['PRON', 'VERB', 'NOUN']),
    (['She', 'teaches', 'math'], ['PRON', 'VERB', 'NOUN']),
    (['We', 'study', 'hard'], ['PRON', 'VERB', 'ADJ'])
]


#train hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)



##new create sentence  ave cumlenin içersinde bulunan her bir sözcüğün türünü etiketle 
test_sentence ="He is a driver".split()

tags=hmm_tagger.tag(test_sentence)

print(f"Yeni cümle : {tags}")


