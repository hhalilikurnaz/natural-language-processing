#bir metni daha küçük parçalara ayırma işidir.
#Önemli bir adım

import nltk #natural language toolkit 
nltk.download("punkt") #metni kelime ve cümle bazında  tokenlara ayırmabilmek için gerekli 

nltk.download("punkt_tab")
text="Hello,World!! How are you ? Hello ..."

#kelime tokenizasyonu : word_tokenize:metni kelimelere ayırır,noktalama işaretleri boşluklar ayrı birer token olarak elde edlecektir 
word_tokens=nltk.word_tokenize(text)
print(word_tokens)

#cümle tokenizasyonu : sent_tokenize: metni cumlelere ayırır.her bir cümle birer token olur 
sentence_tokens=nltk.sent_tokenize(text)
print(sentence_tokens)