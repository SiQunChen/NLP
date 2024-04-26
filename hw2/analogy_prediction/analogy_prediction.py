from gensim.downloader import load
from gensim.models import Word2Vec
import glob

# 加載預訓練的Word2Vec模型
pretrained_model = load('word2vec-google-news-300')

# 讀取資料集
data = []
for folder in ['1_Inflectional_morphology', '2_Derivational_morphology', '3_Encyclopedic_semantics', '4_Lexicographic_semantics']:
    txt_files = glob.glob(f'BATS_3.0/{folder}/*.txt')
    for txt_file in txt_files:
        with open(txt_file) as f:
            data += [line.strip().split('\t') for line in f]

# 訓練自定義Word2Vec模型
custom_model = Word2Vec(data, vector_size=100, window=5, min_count=1, workers=4, epochs=100)
custom_model.train(data, total_examples=len(data), epochs=10)

# 定義函數來進行推理
def analogy(model, a, b, c):
    if isinstance(model, Word2Vec):
        result = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)  # 對於自定義模型使用 wv
    else:
        result = model.most_similar(positive=[b, c], negative=[a], topn=1)  # 對於預訓練模型直接使用
    return result[0][0]

# 測試案例
print("Pretrained Model:")
print("'student', 'students', 'album' =>", analogy(pretrained_model, 'student', 'students', 'album'))
print("'accept', 'accepted', 'apply' =>", analogy(pretrained_model, 'accept', 'accepted', 'apply'))

print("\nCustom Model:")
print("'student', 'students', 'album' =>", analogy(custom_model, 'student', 'students', 'album'))
print("'accept', 'accepted', 'apply' =>", analogy(custom_model, 'accept', 'accepted', 'apply'))