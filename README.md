# Num-Network

A tiny, **numeric** “chatbot” experiment that uses a **single-neuron neural network (logistic regression)** to map **(user_id, message_id) → response_score**. It trains from a CSV (`chat_data.csv`) and then runs an interactive loop where unknown messages can be added with a user-provided **float** “response” value. :contentReference[oaicite:0]{index=0}

> ⚠️ Note: the current `main.py` in the repo is **not valid Python as committed** (imports are on one line; there’s a broken string literal across a newline). This README documents the intended behavior and includes a quick fix snippet you can apply. :contentReference[oaicite:1]{index=1}

---

## What this project is (and isn’t)

### ✅ Is
- A minimal NN demo: **1 neuron**, **sigmoid activation**
- Inputs: **two numeric features**: `user_id` and `message_id`
- Output: a **single float** in `(0, 1)` (sigmoid output)
- Training: basic gradient descent over epochs using squared error (as implemented) :contentReference[oaicite:2]{index=2}

### ❌ Isn’t
- A text-understanding chatbot / LLM
- A model that embeds or parses natural language
- A multi-layer neural network (it’s effectively logistic regression) :contentReference[oaicite:3]{index=3}

---

## Repository contents

- `main.py` — the neural network + CSV loader + interactive loop :contentReference[oaicite:4]{index=4}  
- `chat_data.csv` — training data with columns: `User,Message,Response` :contentReference[oaicite:5]{index=5}  
- `README.md` — currently minimal (“Please enter floats”) :contentReference[oaicite:6]{index=6}  

---

## How it works (technical)

The model is:

- Weights: `w1`, `w2`
- Bias: `b`
- Input vector: `x = [user_id, message_id]`
- Pre-activation: `z = w1*x0 + w2*x1 + b`
- Output: `ŷ = σ(z)` where `σ(z)=1/(1+e^{-z})` :contentReference[oaicite:7]{index=7}

Training loop (as implemented):
- Loss derivative is based on squared error: `dL/dŷ = -2*(y - ŷ)`
- Uses chain rule with `σ'(z)=σ(z)*(1-σ(z))`
- Updates parameters with learning rate `0.1` for `1000` epochs :contentReference[oaicite:8]{index=8}

**Important implication:** because inputs are just IDs (integers), the network is learning a *very rough numeric association* between those IDs and a target float, not linguistic meaning.

---

## Data format: `chat_data.csv`

Header:
```csv
User,Message,Response
