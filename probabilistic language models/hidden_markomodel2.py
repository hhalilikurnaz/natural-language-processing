#import libraries 
import nltk
from nltk.tag import hmm 
from nltk.corpus import conll2000 #kelime etiketleme veri seti


#gerekli veri setini içeri aktar 
nltk.download('conll2000')

train_data=conll2000.tagged_sents('train.txt')
test_data=conll2000.tagged_sents('test.txt')


print(f"train data : {train_data[:1]}")


#train hmm 
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)


#yeni cumle ve test 
test_sentence=" I like going to the cinema .".split()
tags=hmm_tagger.tag(test_sentence)
print(f"test sentence : {tags}")