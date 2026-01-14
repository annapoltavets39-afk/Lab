import re
from collections import Counter
import numpy as np
import pandas as pd

# ================================================
# 1. Зчитування тексту та створення словника частотності
# ================================================
def read_lyrics_csv(file_path):
    """
    Зчитує корпус музичних текстів із CSV-файлу (lyrics.csv)
    та повертає список усіх слів у нижньому регістрі.
    """
    df = pd.read_csv(file_path)
    # Об'єднання всіх текстів у один великий рядок
    text = " ".join(df['lyrics'].dropna().astype(str).tolist())
    # Токенізація (виділення лише слів)
    return re.findall(r'\w+', text.lower())

def get_word_count(words_list):
    """Підрахунок частотності кожного слова."""
    return Counter(words_list)

def get_probabilities(word_counts):
    """Обчислення ймовірностей появи слів у корпусі."""
    total = sum(word_counts.values())
    return {word: count / total for word, count in word_counts.items()}


# ================================================
# 2. Операції редагування (видалення, вставка, заміна, перестановка)
# ================================================
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def insert_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i:] for i in range(len(word) + 1) for c in letters]

def replace_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i+1:] for i in range(len(word)) for c in letters if word[i] != c]

def switch_letter(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]


# ================================================
# 3. Генерація можливих варіантів слова
# ================================================
def edit_one_letter(word, allow_switches=True):
    """Повертає усі слова, що відрізняються від даного на 1 редагування."""
    edits = set()
    edits.update(delete_letter(word))
    edits.update(insert_letter(word))
    edits.update(replace_letter(word))
    if allow_switches:
        edits.update(switch_letter(word))
    return edits

def edit_two_letters(word, allow_switches=True):
    """Повертає усі слова, що відрізняються на 2 редагування."""
    edits = set()
    for e1 in edit_one_letter(word, allow_switches):
        edits.update(edit_one_letter(e1, allow_switches))
    return edits


# ================================================
# 4. Пошук кандидатів на виправлення
# ================================================
def get_candidates(word, vocab, probs, n=1):
    """
    Знаходить найбільш ймовірні варіанти правильного написання.
    Спочатку перевіряє 1-редагування, потім 2-редагування.
    """
    if word in vocab:
        candidates = [word]
    else:
        edits1 = edit_one_letter(word) & vocab
        edits2 = edit_two_letters(word) & vocab
        candidates = edits1 or edits2 or [word]

    # Сортування за ймовірністю (частотністю)
    return sorted([(w, probs.get(w, 0)) for w in candidates],
                  key=lambda x: x[1], reverse=True)[:n]


# ================================================
# 5. Мінімальна відстань редагування (Minimum Edit Distance)
# ================================================
def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    """
    Обчислює мінімальну кількість редагувань для перетворення source → target.
    """
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        D[i][0] = i * del_cost
    for j in range(n + 1):
        D[0][j] = j * ins_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else rep_cost
            D[i][j] = min(
                D[i - 1][j] + del_cost,      # видалення
                D[i][j - 1] + ins_cost,      # вставка
                D[i - 1][j - 1] + cost       # заміна або збіг
            )
    return D[m][n]


# ================================================
# 6. Функція автокорекції
# ================================================
def autocorrect(word, vocab, probs):
    """
    Повертає найімовірніше правильне слово для введеного користувачем.
    """
    suggestions = get_candidates(word, vocab, probs, n=1)
    return suggestions[0][0] if suggestions else word


# ================================================
# 7. Тестування системи автокорекції
# ================================================
if __name__ == "__main__":
    print("=== Завантаження корпусу пісень ===")
    word_list = read_lyrics_csv("lyrics.csv")  # ⚠️ вкажи правильний шлях до файлу
    word_counts = get_word_count(word_list)
    vocab = set(word_counts)
    probs = get_probabilities(word_counts)

    print(f"Кількість унікальних слів у корпусі: {len(vocab)}")

    # Приклади помилкових музичних слів
    test_words = ['beutiful', 'lyfe', 'musick', 'singin', 'lovin']

    print("\n=== Результати автокорекції ===")
    for word in test_words:
        corrected = autocorrect(word, vocab, probs)
        print(f"{word} -> {corrected}")

    print("\n=== Мінімальна відстань редагування ===")
    pairs = [('rhythm', 'rhytym'), ('melody', 'melodie'), ('guitar', 'gutiar')]
    for w1, w2 in pairs:
        dist = min_edit_distance(w1, w2)
        print(f"{w1} -> {w2} : {dist}")
