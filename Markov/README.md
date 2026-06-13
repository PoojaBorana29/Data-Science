# Markov Chain Text Generator
Built from scratch using pure Python — no ML libraries.

---

## What This Is
A text generation engine that learns writing style from any book and generates new text that mimics it. Trained on **The Adventures of Sherlock Holmes**. Same core idea behind modern language models — stripped to its purest form.

---

## How It Works
Reads a book → builds a probability table of which words follow other words → generates new text by sampling from that table one word at a time.

---

## Features
| Function | What It Does |
|---|---|
| `load_file()` | Reads raw book text |
| `clean_text()` | Lowercases, removes punctuation, tokenizes |
| `build_notebook()` | Learns word transition probabilities |
| `generate_text()` | Generates new text from learned patterns |
| `show_learned()` | Shows top words following any word with probability bars |
| `perplexity()` | Measures how well model knows a writing style |

---

## Sample Output

**Generated Text (Order 2)**
```
i should have heard of you we have touched on three sides and on the 
other i rushed out into smoke like so many which present any feature 
of interest i cannot confide it to turn it on the
```

**What Model Learned**
```
After ('always', 'the') →
  woman     50%  ██████████████████████████
  way       50%  ██████████████████████████
```
Model independently discovered "always the woman" is a signature Holmes phrase.

---

## Perplexity Results
Lower = model recognizes the style. Higher = unfamiliar text.
```
Sherlock Holmes text:   3.57
Random sentence:    10457.24

Order 1:  36.39  ← underfit
Order 2:   3.57  ← just right
Order 3:   1.27  ← overfit
```
This directly demonstrates the **bias-variance tradeoff**.

---

## Setup
```
# No external libraries needed — pure Python 3

python markov.py
```

---

## What I Learned
- How language models learn from raw text at the most fundamental level
- Why training and inference must be separate phases
- How frequency naturally becomes probability without any formula
- Why perplexity is a meaningful evaluation metric
- The direct relationship between model complexity and overfitting

---

## Files
```
markov/
  ├── markov.py      ← all code
  ├── markov.txt     ← training data
  └── README.md
```
