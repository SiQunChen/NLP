import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def convert_to_lowercase(text_list):
    return [text.lower() for text in text_list]

# 使用 API 獲取數據
api_url = "https://datasets-server.huggingface.co/first-rows?dataset=carblacac%2Ftwitter-sentiment-analysis&config=default&split=train"
response = requests.get(api_url)
data = response.json()

# 整理數據
texts = [entry['row']['text'] for entry in data['rows']]
labels = [entry['row']['feeling'] for entry in data['rows']]

# 將數據分成訓練集和測試集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_texts = convert_to_lowercase(train_texts)
test_texts = convert_to_lowercase(test_texts)
# print(train_texts, "\n\n\n\n\n", test_texts)

# 將文本向量化
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 使用 Multinomial Naive Bayes 分類器
classifier = MultinomialNB()
classifier.fit(X_train, train_labels)

# 預測測試集
predictions = classifier.predict(X_test)

# 計算 Precision
precision = precision_score(test_labels, predictions, average='weighted')

# 計算 Recall
recall = recall_score(test_labels, predictions, average='weighted')

# 計算 F-measure
f_measure = f1_score(test_labels, predictions, average='weighted')

# 計算 Accuracy
accuracy = accuracy_score(test_labels, predictions)

# 列印評估結果
print("Precision:", precision)
print("Recall:", recall)
print("F-measure:", f_measure)
print("Accuracy:", accuracy)
