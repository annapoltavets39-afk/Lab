# Імпортування бібліотек
import numpy as np
import pandas as pd
import requests
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# --- Завантаження векторів слів для англійської та української ---
print("Завантаження векторів слів для англійської та української...")

# українські fastText вектори (cc.uk.300.vec)
uk_embeddings = KeyedVectors.load_word2vec_format("cc.uk.300.vec", binary=False)

# англійські fastText вектори (cc.en.300.vec)
en_embeddings = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)

print("Вектори успішно завантажені!")

# --- Завантаження словників перекладів EN → UK ---
def load_dict_from_url(url, max_words=100):
    print(f"Завантаження перших {max_words} слів зі словника з {url}...")
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    pairs = [line.split()[:2] for line in lines if len(line.split()) >= 2]
    return dict(pairs[:max_words])

# EN→UK словник Facebook AI (ARRIVAL)
dictionary_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-uk.txt"

translation_dict = load_dict_from_url(dictionary_url)
print(f"Розмір словника перекладів: {len(translation_dict)}")

# --- Побудова матриць X та Y ---
def get_matrices(word_dict, src_embeddings, tgt_embeddings):
    X, Y = [], []
    for src_word, tgt_word in word_dict.items():
        if src_word in src_embeddings and tgt_word in tgt_embeddings:
            X.append(src_embeddings[src_word])
            Y.append(tgt_embeddings[tgt_word])
    return np.array(X), np.array(Y)

# Англійська → Українська
X_train, Y_train = get_matrices(translation_dict, en_embeddings, uk_embeddings)

# --- Обчислення матриці перетворення R ---
def compute_loss(X, Y, R):
    diff = np.dot(X, R) - Y
    return np.sum(diff**2) / len(X)

def compute_gradient(X, Y, R):
    diff = np.dot(X, R) - Y
    return 2 * np.dot(X.T, diff) / len(X)

def align_embeddings(X, Y, steps=100, lr=0.001):
    R = np.random.rand(X.shape[1], Y.shape[1])
    for i in range(steps):
        R -= lr * compute_gradient(X, Y, R)
        if i % 10 == 0:
            print(f"Ітерація {i}, втрата: {compute_loss(X, Y, R):.4f}")
    return R

R = align_embeddings(X_train, Y_train, steps=100, lr=0.05)

# --- Функція перекладу ---
def translate(word, R, src_embeddings, tgt_embeddings):
    if word not in src_embeddings:
        return "Слово відсутнє у словнику"
    transformed_vector = np.dot(src_embeddings[word], R)
    similarities = cosine_similarity(
        transformed_vector.reshape(1, -1), tgt_embeddings.vectors
    )
    closest_idx = similarities.argmax()
    return tgt_embeddings.index_to_key[closest_idx]

# --- Оцінка точності ---
def evaluate(word_dict, R, src_embeddings, tgt_embeddings):
    correct = 0
    for src_word, tgt_word in word_dict.items():
        if src_word in src_embeddings and tgt_word in tgt_embeddings:
            predicted_word = translate(src_word, R, src_embeddings, tgt_embeddings)
            if predicted_word == tgt_word:
                correct += 1
    return correct / len(word_dict)

accuracy = evaluate(translation_dict, R, en_embeddings, uk_embeddings)
print(f"Точність перекладу: {accuracy:.2%}")
