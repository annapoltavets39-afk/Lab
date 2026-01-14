import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ===================== Набор данных страна-столица =====================
# Пример: можно использовать CSV с колонками: country, capital
url_country = "https://raw.githubusercontent.com/datasets/world-countries-capitals/master/data/countries.csv"
data = pd.read_csv(url_country)
data = data[['Country', 'Capital']]
data.columns = ['country', 'capital']

print("Первые 10 строк набора данных (Country-Capital):")
print(data.head(10))

# ===================== Загрузка модели Word2Vec =====================
# Модель должна быть обучена на текстах, где встречаются названия стран и столиц
# Пример: ubercorpus или любая словарная модель с 300d
model = KeyedVectors.load_word2vec_format("ubercorpus.cased.tokenized.word2vec.300d", binary=False)
print("\nМодель успешно загружена!")

# ===================== Косинусная схожесть =====================
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# ===================== Поиск слова для аналогии =====================
def find_relation(word1, relation1, word_embeddings):
    """
    word1 : relation1 :: word2 : ?
    """
    try:
        result = word_embeddings.most_similar(positive=[relation1, word2], negative=[word1], topn=1)
        return result[0][0], result[0][1]
    except KeyError as e:
        # Если слова нет в модели
        return "Невідомо", -1

# ===================== Точность на country-capital =====================
def get_accuracy(word_embeddings, data):
    num_correct = 0
    total = 0
    for i, row in data.iterrows():
        country, capital = row
        if any(w not in word_embeddings.key_to_index for w in [country, capital]):
            continue
        pred, _ = find_relation(country, capital, country, word_embeddings)  # Проверяем аналогию
        if pred == capital:
            num_correct += 1
        total += 1
    return num_correct / total if total > 0 else 0

accuracy = get_accuracy(model, data)
print(f"\nТочність на country-capital: {accuracy:.2%}")

# ===================== Визуализация =====================
def visualize(words, word_embeddings):
    valid_words = [w for w in words if w in word_embeddings.key_to_index]
    vectors = np.array([word_embeddings[w] for w in valid_words])
    pca = PCA(n_components=2)
    comps = pca.fit_transform(vectors)
    plt.figure(figsize=(12, 8))
    plt.scatter(comps[:, 0], comps[:, 1], color='blue')
    for i, w in enumerate(valid_words):
        plt.annotate(w, xy=(comps[i, 0], comps[i, 1]), fontsize=12, color='red')
    plt.title("Визуализация стран и столиц (PCA)")
    plt.show()

# Пример стран для визуализации
words_to_visualize = ["Ukraine", "Kyiv", "France", "Paris", "Germany", "Berlin", "Japan", "Tokyo"]
visualize(words_to_visualize, model)
