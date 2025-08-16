#İngilizce stop words analizi
import nltk 

from nltk.corpus import stopwords
nltk.download("stopwords") #farklı dillerde en çok kullanılan stopwords içeren veri seti 
stop_words_english=set(stopwords.words("english"))
print(stop_words_english)


text="You are very handsome and clever but ther is not enough you should study hard because his can ııyyu who does more handsome "
#textlist lazım
text_list=text.split()
#eğer word ingilizce stop words listesinde yoksa,bu kelimeyi filtrelenmiş listeye ekşiyoruz 
filtered_words=[word for word in text_list if word.lower() not in stop_words_english]
print(filtered_words)



#türkçe stop words analizi
stop_words_turkce=set(stopwords.words("turkish"))
print(stop_words_turkce)
metin="Merhaba ama ben cok yakısıklıyım arkadaslar fakat cok güzel bir ders değil ve size anlatamıyorum veya siz anlamıyorsunuz "
metin_list=metin.split()
filtered_words_tr=[word for word in  metin_list if word.lower() not in stop_words_turkce]
print(filtered_words_tr)
#kütüphanesiz stop word cıkarımı 

#stop word listesi oluştur 
tr_stopwords=["için","bu","ile","mu","mi","özel"]
#örnek türkçe metin
metin="Bu bir denemedir amacımız bu metinde bulunan özel karakterleri elemek mi acaba "

metin_list=metin.split()
filtered_word=[word for word in metin_list if word.lower() not in tr_stopwords]
print(filtered_word)