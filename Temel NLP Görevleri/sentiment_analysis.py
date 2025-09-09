#bir metnin duygusal tonunu tespit ederek kullanıcıların veya musterilerin duygularını anlamaya yardımcı olur.

'''amazon veri seti içerisinde bulunan yorumların positive mi yoksa negatif mi oldugunu sınıflandırmak 
binary classification '''

#import libraries 
import pandas as pd 
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')





#load dataset 
df=pd.read_csv('/Users/halilibrahimkurnaz/Desktop/NLP/Temel NLP Görevleri/data_sentiment.csv')
print(df.head(10))


#text cleaning and preprocessing
lemmatizer=WordNetLemmatizer()
def clean_and_preprocess(text):

    #tokenize 
    tokens=word_tokenize(text.lower())

    #stop words
    filtered_tokens=[token for token in tokens if token not in stopwords.words("english") and token.isalpha()]

    #lemmatization
    lemmatized_tokens=[lemmatizer.lemmatize(token) for token in filtered_tokens ]

    #join words 
    preprocessed_text=' '.join(lemmatized_tokens)

    return preprocessed_text


df['reviewText2']=df['reviewText'].apply(clean_and_preprocess)
#print(df[['reviewText','reviewText2']].head(10))

#sentiment analysis model training
analyzer=SentimentIntensityAnalyzer()

def get_sentiment(text):

    score=analyzer.polarity_scores(text)
    
    sentiment=1 if score["pos"] > 0 else 0 

    return sentiment 


df['sentiment']=df['reviewText2'].apply(get_sentiment)
print(df[['reviewText2','sentiment']].head(10))


#evaluation 

cm=confusion_matrix(df['Positive'],df['sentiment'])
print(f"Confusion Matrix:\n{cm}")

cr=classification_report(df['Positive'],df['sentiment'])
print(f"Classification Report:\n{cr}")

accuracy=accuracy_score(df['Positive'],df['sentiment'])
print(f"Accuracy: {accuracy*100:.2f}%")