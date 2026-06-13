# Project 3: AI Recommendation Logic — Tech Stack Recommender

> **DecodeLabs Industrial Training | Batch 2026 | AI Engineering Track**

---

## 📌 Overview

A content-based filtering recommendation engine built in Python using TF-IDF vectorization and Cosine Similarity. The system maps a user's raw skills and career goals to the most relevant tech job roles — the same mathematical principles powering Netflix, Amazon, and Spotify recommendation engines.

This project marks the shift from **passive classification** (Project 2) to **active prediction** — the system doesn't just label data, it anticipates what a user needs before they fully articulate it.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Skill Intake | Interactive input loop — minimum 3 skills enforced for data density |
| 🔤 Alias Support | Shorthand inputs (`ml`, `py`, `js`) mapped to full skill names |
| ⚙️ TF-IDF Vectorization | Weights specific skills higher than generic ones across all roles |
| 📐 Cosine Similarity | Measures orientation (not magnitude) between user and job role vectors |
| 🏆 Top-N Filtering | Returns Top 3 matches — prevents choice overload |
| 📊 Full Ranking | White-box transparency — all 20 roles shown with scores |
| ❄️ Cold Start Guard | Minimum score threshold detects when no meaningful match exists |
| 🔁 Rerun Loop | Test multiple skill profiles without restarting |

---

##  Architecture

The project follows the **4-Step Ranking Pipeline** from the DecodeLabs briefing:

```
USER INPUT (3+ skills)
        │
        ▼
┌───────────────────┐
│  1. INGESTION     │  Capture user state → skill list → string document
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. SCORING       │  TF-IDF vectorize all roles + user profile
│                   │  → Cosine Similarity against each job role
│                   │  → Score range: 0.0 (no match) to 1.0 (perfect)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. SORTING       │  Sort all roles by similarity score (descending)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. FILTERING     │  Apply MIN_SCORE threshold → truncate to Top-N
│                   │  → Cold Start fallback if no qualified matches
└────────┬──────────┘
         │
         ▼
OUTPUT: Top 3 Recommended Career Paths + Full Ranking Table
```

**Why Cosine Similarity over Euclidean Distance:**
```
Euclidean: measures straight-line distance → sensitive to vector magnitude
           A role with 20 skills unfairly outscore one with 5 identical skills

Cosine:    measures the ANGLE between vectors → magnitude-invariant
           Only cares about the DIRECTION (orientation) of preferences
           Score 1.0 = perfectly aligned  |  0.0 = no shared characteristics
```

**Why TF-IDF over Binary Vectors:**
```
Binary [1,0,1]: treats "python" and "neural_networks" as equally important
                generic skills like "python" appear in 15/20 roles → low signal

TF-IDF:         penalizes skills that appear across many roles (low IDF)
                rewards rare, specific skills (high IDF)
                "neural_networks" → high weight  |  "python" → lower weight
```

---

##  How to Run

**Requirements:** Python 3.x + scikit-learn + pandas

```bash
# 1. Navigate to project folder
cd Project-3

# 2. Create and activate virtual environment

python3 -m venv venv

##Linus
source venv/bin/activate

##Windows
venv\Scripts\activate

# 3. Install dependencies
pip install scikit-learn pandas

# 4. Run — both files must be in the same folder
python3 recommender.py
```

**Example Session:**
```
=======================================================
   DecodeLabs Project 3 — AI Recommendation Engine
   Tech Stack Recommender | Content-Based Filtering
=======================================================
 Dataset loaded: 20 job roles found.

=======================================================
   TECH STACK RECOMMENDER — Skill Intake
=======================================================
  Enter your skills one by one.
  Use underscores for multi-word skills: machine_learning
  Type 'done' when finished (minimum 3 skills).
-------------------------------------------------------
  Skill 1: python
   Added: python
  Skill 2: ml
   Added: machine_learning
  Skill 3: deep_learning
   Added: deep_learning
  Skill 4: done

   Your profile: ['python', 'machine_learning', 'deep_learning']

 TF-IDF prepared: 94 unique skills loaded.

=======================================================
   TOP 3 RECOMMENDED CAREER PATHS
=======================================================

  #1  Machine Learning Engineer
       Match Score : 68.4%
       [#############-------]
       Your matched skills : python, machine_learning, deep_learning

  #2  AI Engineer
       Match Score : 57.1%
       [###########---------]
       Your matched skills : python, deep_learning

  #3  Data Scientist
       Match Score : 43.2%
       [########------------]
       Your matched skills : python, machine_learning
```

---

## 🧠 Concepts Demonstrated

- **Content-Based Filtering** — matches user profile to item attributes, independent of other users
- **TF-IDF Weighting** — Term Frequency × Inverse Document Frequency; penalizes generic, rewards specific
- **Vector Mapping** — qualitative skills transformed into numerical arrays in a shared vocabulary space
- **Cosine Similarity** — magnitude-invariant similarity metric; the industry standard for text matching
- **Cold Start Problem** — handled via MIN_SCORE threshold and onboarding enforcement (min 3 skills)
- **Top-N Filtering** — prevents choice overload by truncating output to highest-scoring matches
- **White-Box Transparency** — full ranking table shows all scores, not just the top results

---

## 📁 Project Structure

```
Project-3/
├── recommender.py      # Main recommendation engine
├── raw_skills.csv      # Dataset: 20 job roles with skill tags
├── README.md           # This file
└── .gitignore
```

---

## 📊 Dataset

`raw_skills.csv` contains 20 tech job roles including:

| Job Role | Key Skills |
|---|---|
| Data Scientist | python, ml, statistics, sql, pandas |
| ML Engineer | python, deep_learning, tensorflow, mlops |
| AI Engineer | neural_networks, nlp, computer_vision, pytorch |
| DevOps Engineer | docker, kubernetes, aws, ci_cd |
| Full Stack Developer | javascript, react, nodejs, sql, apis |
| ... | 15 more roles |

Skills use underscores for multi-word terms (`machine_learning`) so TF-IDF treats them as single tokens.

---


---

##  Author

**Kristan Martinez**
AI Engineering Intern — DecodeLabs Batch 2026

---

