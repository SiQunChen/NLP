import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 開啟 train.txt 和 test.txt 文件
train_data = pd.read_csv(r'C:\Users\user\Desktop\NLP_hw\hw1\train.txt', sep='\t', header=None, names=['label', 'text'])
test_data = pd.read_csv(r'C:\Users\user\Desktop\NLP_hw\hw1\test.txt', sep='\t', header=None, names=['label', 'text'])

# 整理數據
train_texts = train_data['text'].tolist()
train_labels = train_data['label'].tolist()
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# 將文本轉換為小寫
train_texts = [text.lower() for text in train_texts]
test_texts = [text.lower() for text in test_texts]

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
