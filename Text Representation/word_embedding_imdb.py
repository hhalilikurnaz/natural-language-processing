#import libraries
import pandas as pd 
import re 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords


#veri seti yükleme 
df = pd.read_csv("Text Representation/dataset.csv")

documents=df["review"]

#metin temizleme 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)       # URL kaldır
    text = re.sub(r"\d+", "", text)                           # Sayı kaldır
    text = re.sub(r"[^\w\s]", "", text)                       # Noktalama kaldır
    words = text.split()                                      # Kelimelere ayır
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words and len(word) > 2]

    return " ".join(words)                                    # Tekrar birleştir

cleaned_documents=[clean_text(doc) for doc in documents]

#metin tokenization 

tokenized_documents=[simple_preprocess(doc) for doc in cleaned_documents]


#word2vec
model=Word2Vec(sentences=tokenized_documents, vector_size=100, window=5, min_count=1, sg=0)
word_vectors=model.wv

word=list(word_vectors.index_to_key)[:500]  # İlk 500 kelimeyi al
vectors=[word_vectors[word] for word in word]

#clustering KMeans K=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(vectors)
clusters = kmeans.labels_


#PCA 50 >2 
pca= PCA(n_components=2)
reduced_vectors=pca.fit_transform(vectors)


# Sonuçları görselleştirme
plt.figure(figsize=(12, 8))
for i, label in enumerate(word):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], label=label, color='C' + str(clusters[i]))
    plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8, alpha=0.7)
plt.title("Word Embeddings Visualization with KMeans Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid()
plt.show()  



