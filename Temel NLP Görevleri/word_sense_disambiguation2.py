import nltk
nltk.download('wordnet')

nltk.download('punkt')

from pywsd.lesk import simple_lesk, adaptive_lesk, cosine_lesk


sentences = [
    "I go to the bank to deposit money.",
    "The river bank was flooded after the heavy rain."  
]
word="bank"

for s in sentences:
    print(f"Sentence : {s}")

    sense_simple=simple_lesk(s,word)
    print(f"Sense Simple : {sense_simple.definition()}")
    
    print("*****************")

    sense_adaptive=adaptive_lesk(s,word)
    print(f"Sense Adaptive : {sense_adaptive.definition()}")
    print("*****************")

    sense_cosine=cosine_lesk(s,word)
    print(f"Sense Cosine : {sense_cosine.definition()}")
