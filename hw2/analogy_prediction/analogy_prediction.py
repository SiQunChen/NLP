from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 讀取資料集
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # 確保數據分割正確，並去除可能的空白字符
    data = [row.split() for row in df['Word 1'].str.strip() + ' ' + df['Word 2'].str.strip()]
    return data

# 訓練模型
def train_model(data):
    # 增加向量維度到300，窗口大小到5，訓練迭代次數到100以提高學習深度
    model = Word2Vec(sentences=data, vector_size=250, window=1, min_count=1, workers=4, epochs=100)
    model.save("word2vec.model")
    return model

# 載入模型
def load_model(model_path):
    model = Word2Vec.load(model_path)
    return model

# 詞彙相似度估計
def estimate_similarity(model, word1, word2):
    vector1 = model.wv[word1]
    vector2 = model.wv[word2]
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

# 類比預測
def analogy_prediction(model, word1, word2, word3):
    try:
        # 最相似的詞: (word2 - word1) + word3 = word4
        result = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)
        return result[0][0]  # 返回相似度最高的詞
    except KeyError as e:
        return str(e)

# 主函數
def main():
    data = load_dataset('../combined.csv')
    model = train_model(data)
    model = load_model('word2vec.model')
    # 輸入類比詞組
    word1, word2, word3 = "king", "man", "queen"
    result_word = analogy_prediction(model, word1, word2, word3)
    print(f'Given "{word1}" is to "{word2}" as "{word3}" is to X, X might be: "{result_word}"')

if __name__ == "__main__":
    main()
