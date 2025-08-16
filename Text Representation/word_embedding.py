#word embedding kelime gömme =kelimeleri sürekli bir vektör uzayında anlamlı temsil edecek şekilde sayısal vektörlere dönüştürür
#bu temsiller kelimeler arasındaki anlamssal ve dilbilgisel ilişkileri yakalmayı hedefler 
''' Özellikleri ;
Anlamsal Benzerlik > king ve queen kelimeleri benzer vektörler alabilir 
Matematiksel İşlemler > king - man + woman = queen gibi
Kapsamlılık.
Modelleri ;
Word2Vec=Google tarafından geliştirilen kelimeleri vektörlere dönüştüren ve bu vektörleri dildeki ilişkileri yakalayacak şekilde eğiten bir modeldir 
GloVe(Global Vectors for Word Representation) Standofr tarafından geliştirielen kelime gömme temsillerini kelime ortaklıklarını yakalayacak şekilde hesaplayan bir modeldir 
FastText=Facebbok tarafından geliştilen ve kelime gömme temsillerini kelime alt birimlerine de dikkate alarak hesaplayan bir modeldir 

'''


#import libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA #princıble component analysis verideki boyutları azaltır 
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess #>her cümleyi tamamen küçük harflere indirir,noktalamaları atar 




#öernek veri seti oluştur 
sentences=["Köpek çok tatlı bir hayvandır ",
           "Köpekler evcil hayvanlardır",
           "Kediler genellikle bağımsız hareket etmeyi severler",
           "Köpekler sadık ve dost hayvanlardır.",
           "Hayvanlar insanlar için iyi arkadaşlardır.",
           "İnsanlar hayvanlar için iyi bir yol arkadaşıdır"]

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences]

#modeller> word2vec
word2_vec_model=Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)
'''#sentences= eğitim için kullanılan veri seti
#vector size = kelimelerin embedding boyutunu belirler
#window=bir kelimenin bağlamını oluşturan kelimelerin max uzaklığı 
#min_count=eğer bir kelime en az 1 kere gözüküyorsa bu kelime modelde kullanılacaktır 
#sg=modelin mimarisini belirler 0 demek kelimenin çevresindeki kelimelerden kelimeyi tahmin eder demek.eğer 1 olursa sg bu model bir kelimeden çevresindeki kelimeleri tahmin eder 
'''

#>fasttex
fast_text_model=FastText(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)


#sayıllasan metinleri görselleştirme :PCA
def plot_word_embedding(model,title):
    word_vectors=model.wv
    words=list(word_vectors.index_to_key[:1000])
    vectors=[word_vectors[word] for word in words]

    #PCA 
    pca=PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)

    #3d görselleştirme 
    fig=plt.figure(figsize=(12,10))
    ax=fig.add_subplot(111,projection="3d")
    #vektörleri çiz 
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])

    #kelimeleri etiketle 
    for i,word in enumerate(words):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=12)

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC3")
    plt.show()

plot_word_embedding(word2_vec_model,"Word2Vec")
plot_word_embedding(fast_text_model,"FastText")

