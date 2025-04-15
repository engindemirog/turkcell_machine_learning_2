import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = {
    "text":[
        "Kredi borcunuzu hemen ödeyin",
        "Tebrikler kazandınız. Hemen tıklayın.",
        "Yarın toplantıyı unutma",
        "Bedava hediye seni bekliyor",
        "Önemli bir fatura bildirimi var",
        "Bu hafta sonu kahve içelim mi?",
        "Ücretsiz tatil kazandınız!",
        "Bu ay çok çalıştın. Tebrikler!"
    ],
    "label":[1,1,0,1,0,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"]) 
y= df["label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train)
print(X_test)

model = MultinomialNB()
model.fit(X_train,y_train)

y_prediction = model.predict(X_test)

print("Accuracy : ", accuracy_score(y_test,y_prediction) )