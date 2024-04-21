from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 讀取資料集
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    data = [row.split() for row in df['Word 1'] + ' ' + df['Word 2']]
    return data

# 訓練模型
def train_model(data):
    model = Word2Vec(sentences=data, vector_size=100, window=1, min_count=1, workers=4, epochs=100)
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

# 主函數
def main(word1, word2):
    data = load_dataset('../combined.csv')
    model = train_model(data)
    model = load_model('word2vec.model')
    similarity = estimate_similarity(model, word1, word2)
    print(f'"{word1}" 和 "{word2}" 的相似度為: {similarity}')

if __name__ == "__main__":
    main("cat", "delay")
    main("money","dollar")
    main("fuck", "fuck")
    