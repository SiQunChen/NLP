from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def calculate_similarity(file1, file2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([file1, file2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])

def main(file_path1, file_path2):
    with open(file_path1, 'r', encoding='utf-8') as f:
        file1 = f.read()
    with open(file_path2, 'r', encoding='utf-8') as f:
        file2 = f.read()
    similarity = calculate_similarity(file1, file2)
    print(f'文件 "{file_path1}" 和 "{file_path2}" 的相似度為: {similarity[0][0]}')

if __name__ == "__main__":
    main("file1.txt", "file2.txt")
