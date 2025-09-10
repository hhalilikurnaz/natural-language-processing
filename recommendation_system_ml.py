#movie lens veri setini kullanarak basit bir öneri sistemi yapıyoruz 


#import libraries 
from surprise  import Dataset,KNNBasic,accuracy
from surprise.model_selection import train_test_split






#import dataset 
data=Dataset.load_builtin('ml-100k') #(kullanıcı id ,film id ,puan )
print(data)


#train test split 
trainset,testset=train_test_split(data,test_size=0.2,random_state=42)



#ML model create (KNN)

model_options={
    "name": "cosine",
    "user_based":True #kullanıcılar arası benzerlik
}


#train 
model=KNNBasic(sim_options=model_options)
model.fit(trainset)


#test 
predictions=model.test(testset)
accuracy.rmse(predictions)


#recommendation for a user

def get_top_n(predictions,n=10):
    top_n={}
    for uid,iid,true_r,est,_ in predictions:
        if not top_n.get(uid):
            top_n[uid]=[]
        top_n[uid].append((iid,est))


    for uid,user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1],reverse=True)
        top_n[uid]=user_ratings[:n]

        return top_n
    
n=7
top_n=get_top_n(predictions,n)

user_id="2"
print(f"{n} recommendatşon for user   {user_id} film")

for item_id,rating in top_n[user_id]:
    print(f"item id  :{item_id} score  :{rating}")
