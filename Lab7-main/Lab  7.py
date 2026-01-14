import re
import bz2
import os
import math
import urllib.request
from collections import defaultdict, Counter
from nltk.util import ngrams

# === 1. Завантаження та попередня обробка корпусу ===
def download_and_extract_corpus(url, output_file):
    """Завантажує та розпаковує bz2 корпус, якщо його ще немає"""
    compressed_file = output_file + ".bz2"
    if not os.path.exists(output_file):
        print("⬇ Завантаження корпусу UberCorpus...")
        urllib.request.urlretrieve(url, compressed_file)

        print(" Розпакування корпусу...")
        with bz2.open(compressed_file, "rt", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                f_out.write(line)
        print("Корпус готовий:", output_file)
    else:
        print("Корпус уже розпакований:", output_file)


def load_corpus(file_path, max_tokens=200_000):
    """Завантаження та токенізація корпусу"""
    tokens = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            tokens.extend(re.findall(r"\w+", line.lower()))
            if len(tokens) >= max_tokens:
                break
    print(f" Завантажено {len(tokens)} токенів")
    return tokens[:max_tokens]


# === 2. Побудова N-грам моделі ===
def build_ngram_model(tokens, n):
    model = defaultdict(Counter)
    for gram in ngrams(tokens, n):
        prefix, word = tuple(gram[:-1]), gram[-1]
        model[prefix][word] += 1
    return model


# === 3. Прогноз наступного слова ===
def predict_next(model, context, top_k=5):
    context = tuple(context[-(len(next(iter(model))) if model else 0):])
    candidates = model.get(context, {})
    total = sum(candidates.values())
    probs = {word: count / total for word, count in candidates.items()} if total > 0 else {}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]


# === 4. Автозавершення тексту ===
def autocomplete(text, models, top_k=5):
    tokens = text.lower().split()
    for n in reversed(range(2, 6)):
        if len(tokens) >= n - 1:
            context = tokens[-(n - 1):]
            if tuple(context) in models[n]:
                return predict_next(models[n], context, top_k)
    total = sum(models[1].values())
    probs = {word: count / total for word, count in models[1].items()}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]


# === 5. Перплексія (оцінка якості моделі) ===
def perplexity(model, tokens, n):
    N = 0
    log_prob = 0
    for gram in ngrams(tokens, n):
        prefix, word = tuple(gram[:-1]), gram[-1]
        prefix_count = sum(model[prefix].values())
        word_count = model[prefix][word]
        prob = word_count / prefix_count if prefix_count > 0 else 1e-6
        log_prob += math.log(prob)
        N += 1
    return math.exp(-log_prob / N)


# === 6. Основний запуск ===
if __name__ == "__main__":
    url = "https://lang.org.ua/static/downloads/corpora/ubercorpus.tokenized.shuffled.txt.bz2"
    file_path = "ubercorpus.tokenized.txt"

    # Завантаження та розпакування корпусу
    download_and_extract_corpus(url, file_path)

    # Завантаження токенів
    tokens = load_corpus(file_path, max_tokens=200_000)

    # Побудова моделей N-грам
    print("🔧 Побудова моделей N-грам...")
    models = {}
    for n in range(1, 6):
        if n == 1:
            models[n] = Counter(tokens)
        else:
            models[n] = build_ngram_model(tokens, n)

    # Тест автозавершення
    print("\n Тест автозавершення:")
    test_input = "україна є"
    predictions = autocomplete(test_input, models)
    print(f"Введення: '{test_input}'")
    for i, (word, prob) in enumerate(predictions):
        print(f"{i+1}. {word} (ймовірність: {prob:.4f})")

    # Оцінка перплексії
    print("\nы Оцінка перплексії (на 3-грамі):")
    pp = perplexity(models[3], tokens[:10000], 3)
    print(f"Перплексія: {pp:.2f}")
