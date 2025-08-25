#metin üretimi 
""" lstm train with text data 

text data = gpt ile oluştur.(Gündelik hayattan cümleler)






"""
#import libraries
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split




#egitim verisi  olustur chatgpt ile 
texts = {
    "texts": [
        "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum",
        "Fenerbahçe maçlarını izlemek beni çok mutlu ediyor ama aynı zamanda da streslendiriyor",
        "Sabah kahvemi içmeden güne başlayamıyorum",
        "Ders çalışırken lo-fi müzik dinlemek beni odaklıyor",
        "Trafikte sıkışıp kalmak moralimi bozuyor",
        "Kedimle vakit geçirmek bana huzur veriyor",
        "Yeni teknolojiler öğrenmek beni çok heyecanlandırıyor",
        "Kalabalık ortamlarda bazen kendimi huzursuz hissediyorum",
        "Tatilde deniz kenarında kitap okumak en sevdiğim şey",
        "Yağmurlu havalarda dışarı çıkmak yerine film izlemeyi tercih ediyorum",
        "Bugün derste anlattıklarını anlamakta zorlandım",
        "Arkadaşlarımla kahve içmek moralimi düzeltiyor",
        "Bazen sosyal medyada çok fazla vakit harcıyorum",
        "Kütüphanede ders çalışmak bana verimli geliyor",
        "Sabah koşusu yapmak enerjimi artırıyor",
        "Otobüs beklerken çok üşüdüm",
        "Sevdiğim dizinin yeni bölümü çıkmış, çok heyecanlıyım",
        "Kargo çok geciktiği için sinirlendim",
        "Yeni bir proje üzerinde çalışmak beni motive ediyor",
        "Bilgisayarımın yavaşlaması beni çıldırtıyor",
        "Online derslerde dikkatim çok çabuk dağılıyor",
        "Pizzayı kahvaltıda yemek bana keyif veriyor",
        "Arkadaşımın bana sürpriz yapması çok hoşuma gitti",
        "Uzun süre beklemek sabrımı zorluyor",
        "Yeni kitaplar almak beni mutlu ediyor",
        "Gürültülü ortamlarda ders çalışamıyorum",
        "Bugün çok üretken bir gündü",
        "Planlarımın iptal olması moralimi bozuyor",
        "Yeni insanlarla tanışmak bana enerji veriyor",
        "Bazen gelecekle ilgili kaygılarım oluyor",
        "Yürüyüş sırasında podcast dinlemek hoşuma gidiyor",
        "Bugün sporda rekor kırdım, çok gururluyum",
        "Hastanede uzun süre beklemek çok sıkıcıydı",
        "Yeni tarifler denemek mutlu hissettiriyor",
        "Telefonumun şarjı bitince huzursuz oluyorum",
        "Doğada vakit geçirmek bana iyi geliyor",
        "Kötü haberler duyduğumda moralim çok bozuluyor",
        "Bugün çok güzel geri bildirimler aldım",
        "Sınav sonucum beklediğimden düşük geldi",
        "Müze gezmekten inanılmaz keyif alıyorum",
        "Yağmurda ıslanmak beni sinirlendiriyor",
        "Yeni bir dil öğrenmek heyecan verici",
        "İnternetin kesilmesi beni çok rahatsız ediyor",
        "Çocukken oynadığım oyunları hatırlamak mutlu ediyor",
        "Kargomun zamanında gelmesi beni sevindirdi",
        "Gece geç saatlere kadar çalışmak beni yoruyor",
        "Bugün hiç motivasyonum yoktu",
        "Parkta yürüyüş yapmak bana huzur veriyor",
        "Yemek yaparken müzik dinlemeyi seviyorum",
        "Uzun toplantılar beni sıkıyor",
        "Yeni şeyler öğrenmek bana özgüven kazandırıyor",
        "Arkadaşlarımla tartışmak canımı sıkıyor",
        "Birisine yardım etmek beni çok mutlu ediyor",
        "Tatilde dinlenmek bana çok iyi geldi",
        "Sınavlar yaklaştıkça stresim artıyor",
        "Film izlemek kafamı boşaltıyor",
        "Bugün çok fazla işim vardı ama hepsini yetiştirdim",
        "Sabah erken uyanmakta zorlanıyorum",
        "Kendi yaptığım yemeğin güzel olması beni gururlandırıyor",
        "Otobüsün çok kalabalık olması beni bunalttı",
        "Yeni insanlarla tanışmak bana ilham veriyor",
        "Bazen kendime çok yükleniyorum",
        "Kahve dükkanında ders çalışmak hoşuma gidiyor",
        "Bugün çok sakin bir gündü",
        "Arkadaşımın bana hediye alması çok hoşuma gitti",
        "Uykusuz kalmak günümü berbat ediyor",
        "Projelerimde ilerlemek bana motivasyon veriyor",
        "Yolda yürürken güzel bir şarkı duymak beni mutlu etti",
        "Trafikte kavga eden insanlar görmek beni üzüyor",
        "Kamp yapmak bana çok huzur veriyor",
        "Telefonumun bozulması moralimi bozdu",
        "Bugün ailemle vakit geçirmek bana çok iyi geldi",
        "Yağmurdan dolayı dışarı çıkamadım",
        "Sevdiğim kitabı tekrar okumak beni çok mutlu etti",
        "Sabah otobüsü kaçırmak moralimi bozdu",
        "Yıldızları izlemek bana huzur veriyor",
        "Kargo beklemek çok sıkıcı oluyor",
        "Yeni hedefler koymak beni motive ediyor",
        "Arkadaşımın moralinin bozuk olması beni de üzüyor",
        "Bugün güne enerjik başladım",
        "İşlerin üst üste gelmesi beni bunalttı",
        "Çiçeklerle ilgilenmek bana huzur veriyor",
        "Sosyal medyada olumlu yorum almak beni mutlu ediyor",
        "Yemek siparişimin yanlış gelmesi sinirlendirici",
        "Müziğin ritmine kapılmak bana keyif veriyor",
        "Bazen gelecekle ilgili çok heyecanlanıyorum",
        "Bugün çok üretken hissettim",
        "İnternette yaşadığım sorun moralimi bozdu",
        "Deniz kenarında oturmak beni çok rahatlattı",
        "Sınav öncesi kaygı duymak beni geriyor",
        "Başarılarımın takdir edilmesi bana güç veriyor",
        "Bugün işlerim yolunda gitmedi",
        "Kedimin uyurken çıkardığı sesler beni mutlu ediyor",
        "Uzun süre yalnız kalmak beni sıkıyor",
        "Arkadaşlarımla güldüğüm anlar çok kıymetli",
        "Bugün hava çok sıcaktı, dışarı çıkmak istemedim",
        "Çalışmalarımın sonuç verdiğini görmek beni sevindiriyor",
        "Bilgisayar oyunu oynamak bazen çok keyifli oluyor",
        "Bugün hiç verimli olamadım",
        "Kardeşimle vakit geçirmek bana mutluluk veriyor",
        "Yeni projeler üzerinde düşünmek beni heyecanlandırıyor",
        "Yolda çok zaman kaybetmek canımı sıkıyor",
        "Tatilde yeni yerler keşfetmek bana ilham veriyor",
        "Bazen geçmişi çok özlüyorum"
    ]
}




#metin temizleme ve preporcessing:tokenizaton ,padding,label encoding
#tokenization

tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts["texts"]) #metinler üzerindeki kelime frekanslarını öğren ve fit et 
total_words=len(tokenizer.word_index) +1 #kelime sayısı 
print(total_words)

#n- gram dizileri oluştur ve padding uygula 
input_sequences=[]
for text in texts["texts"]:
    #metinleri kelime indexlerine çevir
    token_list=tokenizer.texts_to_sequences([text])[0]


    #her bir metin için n-gram dizileri oluştur
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#en uzun diziyi bulalım ve tüm dizileri aynı uzunluğa getirelim 

max_sequence=max(len(x) for x in input_sequences)

#dizilere padding işlemi uygula , böylece hepsi aynı uzunlukta olacak.
input_sequences=pad_sequences(input_sequences,maxlen=max_sequence,padding="pre")


# X (girdi) ve y (target)
X,y=input_sequences[:,:-1],input_sequences[:,-1] #tüm satırları ve sutunlardan sonuncusuna kadar olanları al sonucnusunu y ye eşitilicem tahmin etmessini isticem 

y=tf.keras.utils.to_categorical(y,num_classes=total_words) #one hot encoding yapıyoruz






#Lstm modeli oluşturma,compile ,train and evaluate
model=Sequential()
model.add(Embedding(total_words,50,input_length=X.shape[1])) #girdi uzunluğu max_sequence

#lstm 
model.add(LSTM(100,return_sequences=False)) # sadece son çıktıyı döndürüyoruz

#output 

model.add(Dense(total_words,activation="softmax")) #çok sınıflı sınıflandırma için softmax kullanıyoruz

#model compile 
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


#model trainigg
model.fit(X,y,epochs=100,verbose=1)


#model prediction


def generate_text(seed_text,next_words):
    #seed text bir başlangıç cümlesi
    #next_words tamamlanacak kelime sayısı 

    for _ in range(next_words):
        #girdi metninii sayısal verilere donustur 
        token_list=tokenizer.texts_to_sequences([seed_text])[0]

        #padding
        token_list=pad_sequences([token_list],maxlen=max_sequence-1,padding="pre")

        #prediction
        predicted_probabilities=model.predict(token_list,verbose=0)

        #en yüksek olasılığa sahip kelimeyi ve indexsini al 
        predicted_index=np.argmax(predicted_probabilities,axis=-1)

        #tokenizer ile kelime indexinden asıl kelime bulunur 
        predicted_words=tokenizer.index_word[predicted_index[0]]

        #tahmin edilen kelimeyi seed_text e ekliyorum 
        seed_text= seed_text + " " + predicted_words

        return seed_text
    

seed_text="Bu hafta sonu "
print(generate_text(seed_text,1)) #1 kelime ekle





