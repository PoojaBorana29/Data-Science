import re
from collections import defaultdict
import random

def load_file(filepath):
    with open(filepath, 'r', encoding = 'utf-8') as f:
        text = f.read()
    return text

text = load_file('markov.txt')
print(text[:200])

def clean_text(text):
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

word = clean_text(text)
print(word[:200])

def build_notebook(words, order):
    notebook = defaultdict(list)
    for i in range(len(words) - order):
        key = tuple(words[i :i+order])
        value = words[i+order]
        notebook[key].append(value)
    return notebook

notebook = build_notebook(word, order=2)
for key, value in list(notebook.items())[:5]:
    print(key, "→", value)


def generate_text(notebook, order, length):
    current_key = random.choice(list(notebook.keys()))
    output = list(current_key)
    
    for i in range(length):
        options = notebook[current_key]
        next_word   = random.choice(options)
        output.append(next_word)
        current_key = tuple(output[-order:])
    return ' '.join(output)

print(generate_text(notebook, order=2, length=100))


from collections import Counter

def show_learned(notebook, word, top_n=5):
    matching = [k for k in notebook.keys() if word in k]
    key = matching[0]
    options = notebook[key]
    counts = Counter(options)
    total = len(options) 
    
    print(f"After {key} →")
    for w, count in counts.most_common(top_n):
        percentage = (counts[w] / total) * 100
        bar = "█" * int(percentage)
        print(f"  {w:<15} {percentage:.0f}%  {bar}")
show_learned(notebook, "holmes")
print("\n")
show_learned(notebook, "watson")
print("\n")
show_learned(notebook, "the")

import math

def perplexity(notebook, words, order):
    total_log_prob = 0
    N = len(words) - order
    
    for i in range(len(words) - order):
        key       = tuple(words[i :i+order])
        next_word = next_word = words[i + order] 
        options   = notebook.get(key, [])
        
        if len(options) == 0:
            continue
            
        counts      = Counter(options)
        total       = total = len(options) 
        probability = counts.get(next_word, 1e-10) / total
        total_log_prob += math.log(probability) 
    
    return math.exp(-1/N * total_log_prob)

# Test on Sherlock Holmes text (trained on same data)
score1 = perplexity(notebook, word, order=2)
print(f"Sherlock Holmes text: {score1:.2f}")

# Test on different words (make a fake sentence)
fake = "the cat sat on the mat and ate the fish".split()
score2 = perplexity(notebook, fake, order=2)
print(f"Random sentence: {score2:.2f}")

for order in [1, 2, 3]:
    notebook = build_notebook(word, order=order)
    score = perplexity(notebook, word, order=order)
    print(f"Order {order} perplexity: {score:.2f}")