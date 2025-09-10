
'''
Kural Tabanlı çeviri
İstatistiksel makine çevirisi 
Nöral makine çevirisi 
> seq2seq modelleri
> transformer modelleri'''

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-tr"  # İngilizce'den Türkçe'ye çeviri modeli
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, how are you?"

# Encode edip modele input olarak vereceğiz 
inputs = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)  # translated değişkenini tanımla

# Translated text metne dönüştürülür 
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")