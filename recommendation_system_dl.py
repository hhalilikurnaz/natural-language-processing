#kullancıılara ilgilerini çekebilecek ürünler,hizmetler veya içerikler önermek için kullanılan sistemlerdir 

''' içerik tabanlı öneri sistemleri  : kullanıcının geçmişte beğendiği ürünlere benzer ürünler önerir
 işbirlikçi filtreleme öneri sistemleri : benzer kullanıcıların beğendiği ürünleri önerir
'''

#problem tanımı 
'''
Recommendation System (Öneri Sistemi) oluşturmak 
user(kullanıcı). - item(urunler ) --- rating(puanlar ).'''



#import libraries 
import numpy as np

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")





#veri setini oluştur 
user_ids=np.array([0,1,2,3,4,0,1,2,3,4]) #kullanıcıların ID'leri 
item_ids=np.array([0,1,2,3,4,1,2,3,4,5]) #urun ID'leri 
ratings=np.array([5,4,3,2,1,4,3,2,1,5]) #verilen puanlar 
 
print(len(user_ids),len(item_ids),len(ratings))


#train test split 

user_ids_train,user_ids_test,item_ids_train,item_ids_test,ratings_train,ratings_test=train_test_split(
    user_ids,item_ids,ratings,test_size=0.2,random_state=42)


#create Neural Network Model 

def create_model(num_users,num_items,embedding_dim):
    #input katmanı
    user_input=Input(shape=(1,),name='user') #kullanıcı ıd 'si girişi
    item_input=Input(shape=(1,),name='item') # urun ıd'si girişi

    #embedding katmanı 

    user_embedding=Embedding(input_dim=num_users,output_dim=embedding_dim,name='user_embedding')(user_input)
    item_embedding=Embedding(input_dim=num_items,output_dim=embedding_dim,name='item_embedding')(item_input)

    #vektorleri duzlestir 

    user_vecs=Flatten()(user_embedding)
    item_vecs=Flatten()(item_embedding)

    dot_product=Dot(axes=1)([user_vecs,item_vecs]) #burada iç çarpım işlemi yapıyoruz düzleştirdiğimiz vektörler arasında 
    output=Dense(1)(dot_product)

    model=Model(inputs=[user_input,item_input],outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),loss="mse")

    return model




                     


#train and test the model

num_users=max(user_ids)+1
num_items=max(item_ids)+1
embedding_dim=8

model=create_model(num_users,num_items,embedding_dim)
model.fit([user_ids_train,item_ids_train],ratings_train,epochs=10,verbose=1,validation_split=0.2)


loss=model.evaluate([user_ids_test,item_ids_test],ratings_test)
print(f"test : {loss}")

#kullanıcı 0 urun 5 için puan tahmini yaıyoruz burada 
user_id=np.array([0])
item_id=np.array([5])
prediction=model.predict([user_id,item_id])
print(f"Predicted rating for user : {user_id[0]} and item : {item_id[0]} is {prediction[0][0]}")