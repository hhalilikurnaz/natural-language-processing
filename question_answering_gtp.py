from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch 


import warnings
warnings.filterwarnings("ignore")

model_name="gpt2"

tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)


def generate_answer(context,question):
    input_text=f"Question: {question}, Context : {context}. Please answer the question based on the context."
    
    inputs=tokenizer.encode(input_text,return_tensors="pt")

    #modeli çalıştırıyorum 
    with torch.no_grad():
        outputs=model.generate(inputs,max_length=500)

        #üretilen yanıtı decode ediyorum 

        answer=tokenizer.decode(outputs[0],skip_special_tokens=True) #cümlelerimizin bitişinde end of sentence (<EOS>,<PAD>) bunları ortadan kaldırıyorum

        #yanıtları ayıklıyorum 

        answer=answer.split("Answer:")[-1].strip()

        return answer
    


context="The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-phase program, it was later expanded under President John F. Kennedy's administration to include a lunar landing."
question="What was the main goal of the Apollo program?"
answer=generate_answer(context,question)
print("Answer:",answer)

context="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for its influence on the philosophy of science. He is best known to the general public for his mass–energy equivalence formula E = mc², which has been dubbed 'the world's most famous equation'."
question="What is Albert Einstein best known for?"
answer=generate_answer(context,question)
print("Answer:",answer)

context="The Great Wall of China is a series of fortifications that stretch across northern China, built to protect against invasions and raids from various nomadic groups. The wall's construction began in the 7th century BC and continued for several dynasties, with the most well-known sections built during the Ming Dynasty (1368–1644). The Great Wall is not a single continuous wall but rather a collection of walls and fortifications that span over 13,000 miles (21,196 kilometers). It is considered one of the most impressive architectural feats in history and is a UNESCO World Heritage site."
question="Why was the Great Wall of China built?"                       
answer=generate_answer(context,question)
print("Answer:",answer)


