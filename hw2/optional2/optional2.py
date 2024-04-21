import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 開啟 train.txt 和 test.txt 文件
train_data = pd.read_csv(r'train.txt', sep='\t', header=None, names=['label', 'text'])
test_data = pd.read_csv(r'test.txt', sep='\t', header=None, names=['label', 'text'])

# 整理數據
train_texts = train_data['text'].tolist()
train_labels = train_data['label'].tolist()
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# 訓練 Word2Vec 模型
# 假設所有的文本已經過分詞處理，如果沒有，需要對中文文本進行分詞
train_tokens = [text.split() for text in train_texts]
word2vec_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)

# 定義一個函數來獲取文本的平均詞向量
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=100):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

# 獲取訓練集和測試集的平均詞向量
X_train = [get_average_word2vec(tokens, word2vec_model.wv, generate_missing=True) for tokens in train_tokens]
X_test = [get_average_word2vec(tokens.split(), word2vec_model.wv, generate_missing=True) for tokens in test_texts]

# 使用高斯 Naive Bayes 分類器
classifier = GaussianNB()
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