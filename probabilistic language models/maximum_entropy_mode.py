#bilinmeyen bir olasılık dağılımını tahmin ederken mümkün oldugunca az varsayımda bulunmayı hedefler.
#bir cümlenin belirli bir sınıfa (poziitf,negatif duygu gibi) ait olma olasılığını tahmin etmek için kullanılabilir.

"""
clasification problem : duygu analizi > olumlu veya olumusz olarak sınıflandıma """


#import libraries 

from nltk.classify import MaxentClassifier


#veri seti tanımlama 

train_data=[
   ({"Love":True,"amazing":True,"great":True,"good":True,"bad":False,"hate":False},'positive'),
   ({"hate":True,"bad":True,"terrible":True,"awful":True,"good":False,"love":False},'negative'),
   ({"good":True,"great":True,"amazing":True,"love":True},'positive'),
   ({"sad":True,"hate":True,"terrible":True},'negative'),
    ({"joy":True,"good":True,"great":True,"love":True},'positive'),]

#train maximum entropy classifier 

classifer= MaxentClassifier.train(train_data, max_iter=100)

#yeni cümle ile test et 
test_sentence="I hate this movie and it was terrible and great"
features={word : (word in test_sentence.lower().split()) for word in ["love", "amazing", "great", "good", "bad", "hate", "terrible", "awful", "sad", "joy"]}
label=classifer.classify(features)
print(f"Result : {label}")
