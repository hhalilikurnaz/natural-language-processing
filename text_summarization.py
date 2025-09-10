# uzun metinlerin daha kısa ve öz bir biçimde özetlenmesini amaçlayan bir doğal dil işleme görevi 

'''
> Özelleştirilmiş (Extractive) Özetleme 
> Özgün (Abstractive) Özetleme
'''

from transformers import pipeline

# Pipeline kullanarak tokenizer falan yapmıyoruz 
summarizer = pipeline("summarization")

text = "The Chrysler Building, the famous art deco New York skyscraper, is turning 90 years old on Friday. The building, which was briefly the world's tallest building before it was surpassed by the Empire State Building in 1931, is known for its distinctive design and ornamentation. It was built by Walter P. Chrysler, the founder of the Chrysler Corporation, and was designed by architect William Van Alen. The building's art deco style is characterized by its use of geometric shapes, bold colors, and intricate details. The Chrysler Building has been featured in numerous films and television shows over the years, and is considered one of the most iconic buildings in New York City."


summary = summarizer(
    text,
    max_length=90,
    min_length=45,
    do_sample=False
)

print(f"Summary: {summary[0]['summary_text']}")

      