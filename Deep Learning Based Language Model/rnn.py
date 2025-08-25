''' Solve Classification problem(Sentiment Analysis) using RNN model (Deep Learning Based Language Model) 
duygu analizi > bir ümlenin etiketlenmesi (positive and negative)
restaurant yorumları degerlendirme 
'''

#import libraries
import numpy as np
import pandas as pd 
from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder






#create dataset
data = {
    "text": [
        "Yemekler çok güzeldi.",
        "Yemekler çok pişmişti.",
        "Garsonlar çok ilgiliydi.",
        "Servis inanılmaz yavaştı.",
        "Tatlılar harikaydı.",
        "Çorba çok tuzluydu.",
        "Ambiyans çok hoştu.",
        "Masanın üstü çok pisti.",
        "Fiyatlar gayet uygundu.",
        "Fiyatlar gereksiz pahalıydı.",
        "Sunum çok şıktı.",
        "Yemek buz gibiydi.",
        "Rezervasyon sorunsuzdu.",
        "Siparişim yanlış geldi.",
        "Personel çok güler yüzlüydü.",
        "Garson kaba davrandı.",
        "Mekan çok konforluydu.",
        "Sandalyeler rahatsızdı.",
        "Lezzetler çok otantikti.",
        "Yemekler bayattı.",
        "Servis çok hızlıydı.",
        "Çatal bıçak kirliydi.",
        "Müzik seçimi çok güzeldi.",
        "Çok gürültülüydü, rahatsız oldum.",
        "Porsiyonlar doyurucuydu.",
        "Porsiyonlar çok küçüktü.",
        "Yemekler tam kıvamındaydı.",
        "Et çok sertti.",
        "İçecekler soğuktu.",
        "İçecekler bayattı.",
        "Dekorasyon modern ve şıktı.",
        "Tuvaletler çok kirliydi.",
        "Garson siparişi çok hızlı aldı.",
        "Siparişimin gelmesi çok uzun sürdü.",
        "Tatlı çeşitleri harikaydı.",
        "Tatlılar bayattı.",
        "Menü çok çeşitliydi.",
        "Menü çok yetersizdi.",
        "Çalışanlar çok profesyoneldi.",
        "Çalışanlar ilgisizdi.",
        "Masa çok rahattı.",
        "Masa dardı ve rahatsızdı.",
        "Garson çok kibar davrandı.",
        "Garson ilgisizdi.",
        "Yemekler taze ve sıcaktı.",
        "Yemekler çok yağlıydı.",
        "Hesap çok makuldü.",
        "Hesap çok şişirilmişti.",
        "Rezervasyon kolaydı.",
        "Rezervasyonum yok sayıldı.",
        "Ortam huzurluydu.",
        "Ortam çok stresliydi.",
        "Çalışanlar samimiydi.",
        "Çalışanlar asık suratlıydı.",
        "Mekan çok ferah.",
        "Mekan çok havasız.",
        "Yemekler mükemmel baharatlıydı.",
        "Yemeklerin tadı yoktu.",
        "Çorba tam kıvamındaydı.",
        "Çorba soğuktu.",
        "Tatlı sunumu çok estetikti.",
        "Tatlı çok şekerliydi.",
        "Fiyat/performans harikaydı.",
        "Fiyatlar kalitesine göre yüksekti.",
        "Servis güler yüzlüydü.",
        "Servis kaba ve soğuktu.",
        "Menü anlaşılırdı.",
        "Menü karmakarışıktı.",
        "Garson önerileri çok iyiydi.",
        "Garson önerisizdi.",
        "Dekorasyon çok yaratıcıydı.",
        "Dekorasyon çok sıradandı.",
        "Mekan temizdi.",
        "Mekan hijyensizdi.",
        "Yemekler çok doyurucuydu.",
        "Yemekler az pişmişti.",
        "Tatlı çok hafifti.",
        "Tatlı çok ağırdı.",
        "Masa düzeni harikaydı.",
        "Masa düzeni kötüydü.",
        "Çalışanlar çok bilgiliydi.",
        "Çalışanlar bilgisizdi.",
        "Ortam romantikti.",
        "Ortam çok sıkıcıydı.",
        "İçecekler taptazeydi.",
        "İçecekler bayattı.",
        "Servis zamanında geldi.",
        "Servis geç geldi.",
        "Tatlılar çok lezizdi.",
        "Tatlılar lezzetsizdi.",
        "Garson çok anlayışlıydı.",
        "Garson kaba ve ters davrandı.",
        "Yemekler tam istediğim gibiydi.",
        "Yemekler beklentimin altındaydı.",
        "Ortam çok konforluydu.",
        "Ortam çok rahatsızdı.",
        "Hizmet kusursuzdu.",
        "Hizmet berbattı.",
        "Sunum çok şık ve özenliydi.",
        "Sunum özensizdi."
        
    ],
    "label": [
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative",
        "positive","negative","positive","negative","positive","negative","positive","negative","positive","negative"
    ]
}

print(len(data["text"]))
print(len(data["label"]))


df=pd.DataFrame(data)
print(df.head(10))


#metin temizleme işlemleri  ve preprocessing işlemleri(#tokenization,padding,label encoding işlemleri)
 #tokenization 
tokenizer=Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences=tokenizer.texts_to_sequences(df["text"])
word_index=tokenizer.word_index


#padding process
#data frame içinde farklı cümleler var cümleler içinde kelime sayıları farklı bunun için padding işlemi yapıyoruz 

maxlen=max(len(seq) for seq in sequences)
X=pad_sequences(sequences,maxlen=maxlen)
print(X.shape)


#label encoding (veri seti içindeki labelların encode edilmesi)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(df["label"])

#train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)




#metin temsili: word embedding işlemleri : word2vec 
sentences=[text.split() for text in df["text"]]
word2vec_model=Word2Vec(sentences,vector_size=50,window=5,min_count=1)

embedding_dim=50
embedding_matrix=np.zeros((len(word_index) + 1,embedding_dim))

for word,i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]=word2vec_model.wv[word]
        

#modelling: build,train and test rnn model 

model=Sequential()

#embedding
model.add(Embedding(input_dim=len(word_index)+1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))
                    

#rnn layer 
model.add(SimpleRNN(units=50,return_sequences=False))


#outout layer 

model.add(Dense(units=1,activation='sigmoid'))

#conpiling the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#train model 
model.fit(X_train,y_train,epochs=10,batch_size=2,validation_data=(X_test,y_test))







#evaluation:
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test loss : {loss}")
print(f"Accuracy : {accuracy}")


#cümle sınıflandırma calısması 

def classify_sentence(sentence):
    seq=tokenizer.texts_to_sequences([sentence])
    padded_seq=pad_sequences(seq,maxlen=maxlen)
    
    prediction=model.predict(padded_seq)
    predicted_class=(prediction> 0.5).astype(int)
    
    label="positive" if predicted_class[0][0] == 1 else "negative"
    
    return prediction 

sentence="Restaurant çok güzeldi ve temizdi yemeklerde öyle çok gğüzeldi"
result=classify_sentence(sentence)
print(f"Result : {result}")