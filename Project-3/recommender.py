# ============================================================
# DecodeLabs AI Internship — Project 3
# AI Recommendation Logic: Tech Stack Recommender
# Kristan Martinez | Batch 2026
# ============================================================
# Pipeline: Input → TF-IDF Vectorization → Cosine Similarity
#           → Sort → Top-N Filter → Output
# ============================================================

try:
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    raise ImportError(
        "Required libraries missing. Install with: pip install scikit-learn pandas"
    ) from exc

import os


TOP_N       = 3       # Number of recommendations to show
MIN_SCORE   = 0.05    # Minimum similarity score to qualify (cold start filter)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "raw_skills.csv")

ALIASES = {
    "ml": "machine_learning",
    "ai": "artificial_intelligence",
    "js": "javascript",
    "py": "python",
}


# PHASE 1: INPUT — Load Dataset & Ingest User Profile 

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'\n"
            "Make sure raw_skills.csv is in the same folder as this script."
        )

    df = pd.read_csv(filepath)

    required = {"job_role", "skills"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    print(f" Dataset loaded: {len(df)} job roles found.")
    return df

def get_user_skills() -> str:
    """
    Ingestion Step — capture user state.
    Requires minimum 3 skills for sufficient data density.
    Returns skills as a single space-separated string (matches TF-IDF format).
    """
    print("\n" + "=" * 55)
    print("   TECH STACK RECOMMENDER — Skill Intake")
    print("=" * 55)
    print("  Enter your skills one by one.")
    print("  Use underscores for multi-word skills: (Example) machine_learning")
    print("  Type 'done' when finished (minimum 3 skills).")
    print("-" * 55)

    skills = []
    while True:
        skill = input(f"  Skill {len(skills)+1}: ").strip().lower().replace(" ", "_")
        skill=ALIASES.get(skill, skill)  # Map aliases to abrivated forms
        
        if skill == "done":
            if len(skills) < 3:
                print(f" !!  Need at least 3 skills. You have {len(skills)}. Keep going!")
                continue
            break
        elif skill == "":
            continue
        elif skill in skills:
            print(f" !!  '{skill}' already added. Try a different one.")
        else:
            skills.append(skill)
            print(f"   Added: {skill}")

    print(f"\n   Your profile: {skills}")
    return " ".join(skills)   

# PHASE 2: PROCESS — TF-IDF + Cosine Similarity 

def prepare_vectorizer(df: pd.DataFrame):
    """
    Processing Step — TF-IDF vectorization.
    Converts job role skills into a matrix of TF-IDF features.
        - Each row = a job role
        - Each column = a unique skill
        - Cell value = importance of that skill for the role (0 to 1)
    """
    vectorizer = TfidfVectorizer()

    job_vectors = vectorizer.fit_transform(df["skills"])

    vocab_size = len(vectorizer.vocabulary_)

    print(
                f"\n TF-IDF prepared:"
                f" {vocab_size} unique skills loaded."
    )

    return vectorizer, job_vectors

def vectorize_user(vectorizer, user_profile: str):
    return vectorizer.transform([user_profile])

def score_and_rank(df: pd.DataFrame, job_vectors, user_vector) -> pd.DataFrame:
    """
    Scoring + Sorting Steps.
    Calculates cosine similarity between user vector and every job role.
    Cosine similarity measures ANGLE (orientation) not magnitude —
    so a role with 20 skills won't unfairly outscore one with 5.

    Score 1.0 = perfect alignment
    Score 0.0 = no shared characteristics
    """
    # Calculate cosine similarity — returns array of shape (1, n_jobs)
    scores = cosine_similarity(user_vector, job_vectors).flatten()

    # Attach scores to dataframe and sort descending
    df = df.copy()
    df["similarity_score"] = scores
    df_sorted = df.sort_values("similarity_score", ascending=False)

    return df_sorted

#  PHASE 3: OUTPUT — Filter & Display Top-N Results 

def display_recommendations(df_sorted: pd.DataFrame, user_profile: str) -> None:
    """
    Filtering Step — prevent choice overload.
    Truncates to Top-N, applies minimum score threshold (cold start guard).
    """
    print("\n" + "=" * 55)
    print(f"   TOP {TOP_N} RECOMMENDED CAREER PATHS")
    print("=" * 55)

    # Filter by minimum score (cold start protection)
    qualified = df_sorted[df_sorted["similarity_score"] >= MIN_SCORE]

    if qualified.empty:
        # Cold Start fallback — no meaningful matches found
        print("\n  !!  Cold Start Detected!")
        print("  Your skills don't match any role closely enough.")
        print("  Try adding more specific technical skills.")
        return

    # Take Top-N
    top_results = qualified.head(TOP_N)

    for rank, (_, row) in enumerate(top_results.iterrows(), start=1):
        score_pct  = row["similarity_score"] * 100
        role       = row["job_role"]
        role_skills = row["skills"].replace("_", " ").split()

        # Find matching skills between user and role
        user_skills_set = set(user_profile.split())
        role_skills_set = set(row["skills"].split())
        matched = [s.replace("_", " ") for s in user_skills_set & role_skills_set]

        # Score bar visualisation
        bar_length = int(score_pct / 5)
        bar = "#" * bar_length + "-" * (20 - bar_length)

        print(f"\n  #{rank}  {role}")
        print(f"       Match Score : {score_pct:.1f}%")
        print(f"       [{bar}]")
        print(f"       Your matched skills : {', '.join(matched) if matched else 'indirect match'}")

    # Show all scores for transparency (white-box principle)
    print(f"\n{'-'*55}")
    print("   Full Ranking (all roles):")
    print(f"{'-'*55}")
    
    top_results = qualified.head(TOP_N)

    cutoff = top_results.iloc[-1]["similarity_score"]
    
    for _, row in df_sorted.iterrows():
        score_pct = row["similarity_score"] * 100
        marker = (" <-"if row["similarity_score"] >= cutoff else "" )
        print(f"  {row['job_role']:<30} {score_pct:5.1f}%{marker}")

    print(f"\n   Tip: Add more specific skills to improve match accuracy.")



def main():
    print("=" * 55)
    print("   DecodeLabs Project 3 — AI Recommendation Engine")
    print("   Tech Stack Recommender | Content-Based Filtering")
    print("=" * 55)

    # Load dataset
    df = load_dataset(DATA_FILE)
    vectorizer, job_vectors = prepare_vectorizer(df)
    
    # Run recommendation loop
    while True:
        # PHASE 1: Ingestion
        user_profile = get_user_skills()

        # PHASE 2: TF-IDF + Cosine Similarity
        user_vector = vectorize_user(vectorizer, user_profile)
        df_scored = score_and_rank(df, job_vectors, user_vector)

        # PHASE 3: Top-N Output
        display_recommendations(df_scored, user_profile)

        # Ask to run again
        print(f"\n{'='*55}")
        again = input("  Try with different skills? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Recommendation engine shut down. Good luck! \n")
            break

if __name__ == "__main__":
    main()