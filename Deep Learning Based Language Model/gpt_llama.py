'''
metin üretimi
gpt2-
llama
'''

#import libraries 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer,AutoModelForCausalLM


#modelin tanımlanması 
model_name="gpt2"
llama_model_name="huggyllama/llama-7b"



#tokenizer tanımlama  ve model olusturma 
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama=AutoTokenizer.from_pretrained(llama_model_name)


model=GPT2LMHeadModel.from_pretrained(model_name) #gpt2 yi model değişkenine atadık
llama_model=AutoModelForCausalLM.from_pretrained(llama_model_name)




#metin üretimi için gerekli başlangıç texti
input_text="Deep learning is a subset of machine learning that focuses on"


#tokenization
inputs=tokenizer.encode(input_text,return_tensors="pt")
inputs_llama=tokenizer_llama(input_text,return_tensors="pt")

#metin üretimi
output=model.generate(inputs,max_length=15)
outpusts_llama=llama_model.generate(inputs_llama.input_ids,max_length=55)

#modelin ürettiği tokenları okunabilir hale getiriyoruz
generated_text=tokenizer.decode(output[0],skip_special_tokens=True)#özel tokenları metinden çıkar (başlangıç bitiş tokenları gibi)
generated_text_llama=tokenizer.decode(outpusts_llama[0],skip_special_tokens=True)
print(generated_text)
print(generated_text_llama)
