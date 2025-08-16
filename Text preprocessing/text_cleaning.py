#Metinlerde bulunan fazla boşlukları ortadan kaldır.
text="Hello,   World.   2026"
a=text.split()
print(a)
cleaned_text1="".join(a)
print(cleaned_text1)

#Büyük harfleri küçük harflere çevir
text="Hello,WORLd ! 2035"
clenaed_text2=text.lower()
print(clenaed_text2) #küçük harfe çevir 

#Noktalama işaretlerini kaldır 
import string
text="Hello , Wordl ! 2035"
cleaned_text3=text.translate(str.maketrans("", "", string.punctuation)) #bu ne demek .Bizim buradaki ilk 2 argumanımız boş hiçbir karakteri hiçbir karakterle değiştirmeden sadece noktalama işaretlerini kaldırıyoruz 
print(f"text :{text} \n cleaned_text3 :{cleaned_text3}")

#Özel karakterleri kaldır 
import re #regular expression kütüphanemiz düzenli ifadelerle calısır 
text="Helloo #### Wordl !!! Welcome to 2026 !!!!!$$½££>#£>$"

cleaned_text4=re.sub(r"[^A-Za-z0-9]","",text) #içinde A-Z arası a-z arası ve rakam ara diğerlerini umursama demek oluyor aslında
print(f"text : {text} \n cleaned_text4 : {cleaned_text4}")
#sub subtract çıkar demek zaten



#yazım hatalarını düzelt
#textblob kullanıcaz 
from textblob import TextBlob #metin analizlerinde kullanılan bir kütüphane 
text="Helloo Wlrod! 2035 "
clenaed_text5=TextBlob(text).correct() # correct : Yazım hatalarını düzeltir
print(f" text : {text} cleaned_text5 : {clenaed_text5}")

#html ya da url etiketlerini kaldır 
#beautiful soup kullanıcaz 
from bs4 import BeautifulSoup
html_text="<div>Hello,World! 2035</div>" #html etiketi var 
cleaned_text6=BeautifulSoup(html_text,"html.parser").get_text() #beautiful soup ile html yapisini parse et ,get text ile text kısmını çek 
print(f"text : {html_text} cleaned_text6 : {cleaned_text6}")