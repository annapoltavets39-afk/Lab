import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import random

# Завантаження корпусу
nltk.download('brown')
nltk.download('universal_tagset')  # для стандартизованих тегів

# ================================================
# 1. Зчитування корпусу Brown (у спрощених тегах)
# ================================================
# Використаємо universal_tagset (NOUN, VERB, ADJ, ADV...)
tagged_sents = list(brown.tagged_sents(tagset="universal"))

# Перемішування для стабільності
random.seed(42)
random.shuffle(tagged_sents)

# Розділення train/test
split_point = int(0.8 * len(tagged_sents))
train_sents = tagged_sents[:split_point]
test_sents = tagged_sents[split_point:]

# ================================================
# 2. Побудова частот переходів і емісій
# ================================================
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

for sentence in train_sents:
    prev_tag = '<s>'
    for word, tag in sentence:
        word = word.lower()

        transition_counts[prev_tag][tag] += 1
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

        prev_tag = tag

    transition_counts[prev_tag]['</s>'] += 1

# ================================================
# 3. Обчислення ймовірностей переходів (A) та емісій (B)
# ================================================
A = defaultdict(dict)
for prev_tag in transition_counts:
    total = sum(transition_counts[prev_tag].values())
    for tag in transition_counts[prev_tag]:
        A[prev_tag][tag] = transition_counts[prev_tag][tag] / total

B = defaultdict(dict)
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    for word in emission_counts[tag]:
        B[tag][word] = emission_counts[tag][word] / total

# ================================================
# 4. Алгоритм Вітербі
# ================================================
def viterbi(sentence, A, B, all_tags):
    V = [{}]
    path = {}

    # Ініціалізація
    for tag in all_tags:
        trans_p = A['<s>'].get(tag, 1e-6)
        emis_p = B[tag].get(sentence[0], 1e-6)
        V[0][tag] = trans_p * emis_p
        path[tag] = [tag]

    # Рекурсія
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for curr_tag in all_tags:
            emis_p = B[curr_tag].get(sentence[t], 1e-6)
            best = []

            for prev_tag in all_tags:
                trans_p = A[prev_tag].get(curr_tag, 1e-6)
                prob = V[t-1].get(prev_tag, 0) * trans_p * emis_p
                best.append((prob, prev_tag))

            prob, prev_tag = max(best)
            V[t][curr_tag] = prob
            new_path[curr_tag] = path[prev_tag] + [curr_tag]

        path = new_path

    # Завершення
    n = len(sentence) - 1
    (prob, final_tag) = max((V[n][tag], tag) for tag in all_tags)
    return path[final_tag]

# ================================================
# 5. Оцінка точності HMM моделі
# ================================================
all_tags = list(tag_counts.keys())

def evaluate(test_sents, A, B, all_tags):
    total = 0
    correct = 0

    for sentence in test_sents:
        words = [w.lower() for w, t in sentence]
        true_tags = [t for w, t in sentence]

        predicted = viterbi(words, A, B, all_tags)

        for p, t in zip(predicted, true_tags):
            if p == t:
                correct += 1
            total += 1

    return correct / total

accuracy = evaluate(test_sents, A, B, all_tags)
print(f"Accuracy of HMM model (Brown corpus): {accuracy:.4f}")

# ================================================
# 6. Порівняння з базовим NLTK POS-теггером
# ================================================
def nltk_pos_accuracy(test_sents):
    total = 0
    correct = 0

    for sentence in test_sents:
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]

        predicted = nltk.pos_tag(words, tagset="universal")
        predicted_tags = [t for w, t in predicted]

        for p, t in zip(predicted_tags, true_tags):
            if p == t:
                correct += 1
            total += 1

    return correct / total

nltk_acc = nltk_pos_accuracy(test_sents)
print(f"Accuracy of NLTK POS-tagger on Brown corpus: {nltk_acc:.4f}")
