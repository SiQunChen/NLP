import requests
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_text(text_list):
    processed_texts = []
    for text in text_list:
        # 轉換為小寫
        text = text.lower()
        # 移除非字母字符
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        processed_texts.append(text)
    return processed_texts

# 使用 API 獲取數據
api_url = "https://datasets-server.huggingface.co/first-rows?dataset=carblacac%2Ftwitter-sentiment-analysis&config=default&split=train"
response = requests.get(api_url)
data = response.json()

# 整理數據
texts = [entry['row']['text'] for entry in data['rows']]
labels = [entry['row']['feeling'] for entry in data['rows']]

# 將數據分成訓練集和測試集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 文本預處理
train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)

# 使用 TfidfVectorizer 向量化文本
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 使用隨機森林分類器
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, train_labels)

# 預測測試集
predictions = classifier.predict(X_test)

# 計算 Accuracy
accuracy = accuracy_score(test_labels, predictions)

# 計算 Precision
precision = precision_score(test_labels, predictions, average='weighted')

# 計算 Recall
recall = recall_score(test_labels, predictions, average='weighted')

# 計算 F-measure
f_measure = f1_score(test_labels, predictions, average='weighted')

# 列印評估結果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F-measure:", f_measure)
