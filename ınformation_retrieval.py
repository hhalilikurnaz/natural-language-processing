#kullanıcının sorduğu soruya en iyi yanıtı getiemeye calısan sistem 

#import libraries 

from transformers import BertTokenizer,BertModel

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

#tokenizer and model create 
model_name="bert-base-uncased" #kucuk boyutlu bert modeli 
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name) #önceden eğitilmiş bert modeli 







#veri olustur :karşılaştıracak belgeleri ve sorgu cümlemizi oluşturacğaız 


documents=[
    "Machine Learning is a field of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data.",
    "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
    "Deep Learning is a subset of machine learning that uses neural networks with many layers to model and understand complex patterns in data.",
    "Information Retrieval is the process of obtaining relevant information from a large repository, such as a database or the internet, based on a user's query.",
    "I go to shop "
]

query="What is Deep Learning  ?"






#bert ile bilgi getirimi 
def get_embedding(text):
    inputs=tokenizer(text,return_tensors="pt",truncation=True,padding=True)

    #modeli çalıştrı 
    outputs=model(**inputs)

    #son gizli katmanı alalım 
    last_hidden_state=outputs.last_hidden_state

    #metn temsili oluştur 
    embedding=last_hidden_state.mean(dim=1)

    #vektoru numpy oalrak return et 
    return embedding.detach().numpy()






#belgeler ve sorgular için embedding vektorlerini al 

doc_embeddings=np.vstack([get_embedding(doc) for doc in documents])

query_embedding=get_embedding(query)

#cosine similarity ile belgeler arasında benzerliği hesaplayaım 

similarities=cosine_similarity(query_embedding,doc_embeddings)

#her belgenin benzelrik skoru 

for i,score in enumerate(similarities[0]):
    print(f"Document {documents[i]}: \n{score}")


#en yuksek benzerlik skoruna sahip belgenin indexi 
most_similar_index=similarities.argmax()
print(f"Most similar document : {documents[most_similar_index]}")