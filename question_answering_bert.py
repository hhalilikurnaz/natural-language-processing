#amaç soruyu anlamak ve uygun cevabı bulmak cevabıda doğal dille vermek 


from transformers import BertTokenizer,BertForQuestionAnswering 
import torch

import warnings
warnings.filterwarnings("ignore")

#squad veri seti üzerinde eğitilmiş model(fine tuning yapılmış)
model_name="bert-large-uncased-whole-word-masking-finetuned-squad"

#bert tokenizer
tokenizer=BertTokenizer.from_pretrained(model_name)     

#soru cevaplama görevi için ince ayar yapılmış bert modeli  
model=BertForQuestionAnswering.from_pretrained(model_name)

#cevapları tahmin eden fonksiyon 

def predict_answer(context,question):

    """
    context=metin
    question=soru 
    amac: metin içerisinden soruyu bulmak 
    
    
    1) tokenize 
    2)metin icerisinde soruyu ara 
    3) metin icersiinde sorunun ceavbının nerelerde olabileceginin skorlarını hesapla
    4)skorlardan tokenların indexlerini hesapladık 
    5)tokenları bulduk yani ceavbı bulduk 
    6)okubnab ilir olmasi için tokanları stringe çevirddim 
    """

    #metni ve soruyu tokenlara ayiralım ve modele uygun hale getirelim 
    encoding=tokenizer.encode_plus(question,context,return_tensors="pt",max_length=512,truncation=True)
    #truncation demek token sayısı 512yi geçerse kesilsin demek 

    #giriş tensorlerini hazırla 
    input_ids=encoding["input_ids"]
    attention_mask=encoding["attention_mask"] #hangi tokenlerin dikkate alınacağını belirler 


    #modeli çalıştır ve skorları hesapla 

    with torch.no_grad():
        start_scores,end_scores=model(input_ids,attention_mask=attention_mask,return_dict=False)

        #en yuksek olasılağa sahip start ve end indexlerini hesaplayalım 
        start_index=torch.argmax(start_scores,dim=1).item()
        end_index=torch.argmax(end_scores,dim=1).item()


        #token idlerini kullanarak cevap metnini elde edelim
        answer_tokens=tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])

        #tokenları birleştir ve okunabilir hale getir
        answer=tokenizer.convert_tokens_to_string(answer_tokens)

        return answer 
    
question="What is the capital of Turkey ?"
context="Ankara is the capital of Turkey. It is known for its rich history and cultural heritage."

answer=predict_answer(context,question)
print(answer)