#  Project 1: Rule-Based AI Chatbot

> **DecodeLabs Industrial Training | Batch 2026 | AI Engineering Track**

---

## 📌 Overview

A rule-based conversational AI chatbot built entirely from first principles in Python — no external libraries, no machine learning frameworks. This project demonstrates mastery of the **IPO Model** (Input → Process → Output), deterministic logic design, and the foundational architecture that underlies modern AI guardrail systems.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Exact Match Engine | O(1) dictionary lookup — faster than if-elif chains |
| 🧩 Keyword Match Engine | Partial input scanning for natural phrasing |
| 🕒 Dynamic Responses | Live time & date via lambda functions |
| 🧹 Input Sanitization | Handles case, whitespace, and empty input |
| 📜 Conversation History | Timestamped log of every exchange |
| 🚪 Clean Exit | Graceful shutdown via `exit` command |

---

## 🏗️ Architecture

The chatbot follows a **3-Tier Response Engine**:

```
User Input
    │
    ▼
[ Sanitization ]  →  lower().strip()
    │
    ▼
[ Tier 1 ]  →  Exact dictionary match       O(1)
    │ (no match)
    ▼
[ Tier 2 ]  →  Keyword scan                 O(k)
    │ (no match)
    ▼
[ Tier 3 ]  →  Fallback response
    │
    ▼
Output to User + Log to History
```

This mirrors the **AI Guardrail architecture** used in production systems like NVIDIA NeMo and Llama Guard — where a deterministic rule layer sits above a probabilistic core.

---

## 🚀 How to Run

**Requirements:** Python 3.x — no installs needed.

```bash
# Clone the repository
git clone https://github.com/<your-username>/DecodeLabs-Internship.git

# Navigate to the project folder
cd DecodeLabs-Internship/Project-1-Rule-Based-Chatbot

# Run the chatbot
python Rule-Based AI Chatbot.py
```

**Example session:**
```
====================================================
   DecodeLabs DBot v2.0 — Rule-Based AI Chatbot
   Type 'exit' to quit | 'history' to review chat
====================================================

You: hello
Bot: Hey there! How can I help you today?

You: what is ai
Bot: AI is the simulation of human intelligence by machines — and you're learning to build it!

You: what time is it
Bot: The current time is 02:30 PM.

You: exit
Bot: Shutting down. Great work today! 👋
```

---

## 🧠 Concepts Demonstrated

- **Control Flow & Decision Logic** — the foundation of all AI systems
- **Hash Map / Dictionary Lookup** — O(1) vs O(n) algorithmic efficiency
- **Input Sanitization & Normalization** — core data preprocessing
- **IPO Model** — Input/Process/Output system design
- **White-Box AI** — fully traceable, zero hallucination risk
- **Lambdas & Callable Values** — dynamic response generation

---

## 📁 Project Structure

```
Project-1-Rule-Based-Chatbot/
│
├── chatbot.py       # Main chatbot source code
└── README.md        # This file
```

---


