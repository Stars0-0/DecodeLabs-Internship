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
DATA_FILE   = "raw_skills.csv"

# PHASE 1: INPUT — Load Dataset & Ingest User Profile 

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load job roles and their skill sets from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: '{filepath}'\n"
            "Make sure raw_skills.csv is in the same folder as this script."
        )
    df = pd.read_csv(filepath)
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

def build_tfidf_matrix(df: pd.DataFrame, user_profile: str):
    """
    Vector Mapping Step.
    TF-IDF transforms all job role skill strings + user profile
    into weighted numerical vectors in a shared vocabulary space.

    CRITICAL: user profile must be vectorized in the SAME space
    as the job roles — fitting on all documents together ensures
    the vocabulary is shared.
    """
    # Combine job role skill strings with user profile
    all_documents = list(df["skills"]) + [user_profile]

    # TF-IDF Vectorizer
    # - Penalizes generic skills that appear in every role (low IDF)
    # - Rewards specific skills unique to fewer roles (high IDF)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents).tocsr()

    # Separate job role vectors from user vector
    job_vectors  = tfidf_matrix[:-1]    # All rows except last = job roles
    user_vector  = tfidf_matrix[-1]    # Last row = user profile
    vocab_size = len(vectorizer.vocabulary_)
    print(f"\n  TF-IDF vectorized: {vocab_size} unique skills in vocabulary")
    print(f"   Each job role → vector of {vocab_size} weighted dimensions")

    return job_vectors, user_vector

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
        bar = "| " * bar_length + " :" * (20 - bar_length)

        print(f"\n  #{rank}  {role}")
        print(f"       Match Score : {score_pct:.1f}%")
        print(f"       [{bar}]")
        print(f"       Your matched skills : {', '.join(matched) if matched else 'indirect match'}")

    # Show all scores for transparency (white-box principle)
    print(f"\n{'-'*55}")
    print("   Full Ranking (all roles):")
    print(f"{'-'*55}")
    for _, row in df_sorted.iterrows():
        score_pct = row["similarity_score"] * 100
        marker = " <-" if score_pct >= df_sorted.iloc[TOP_N-1]["similarity_score"] * 100 else ""
        print(f"  {row['job_role']:<30} {score_pct:5.1f}%{marker}")

    print(f"\n   Tip: Add more specific skills to improve match accuracy.")



def main():
    print("=" * 55)
    print("   DecodeLabs Project 3 — AI Recommendation Engine")
    print("   Tech Stack Recommender | Content-Based Filtering")
    print("=" * 55)

    # Load dataset
    df = load_dataset(DATA_FILE)

    # Run recommendation loop
    while True:
        # PHASE 1: Ingestion
        user_profile = get_user_skills()

        # PHASE 2: TF-IDF + Cosine Similarity
        job_vectors, user_vector = build_tfidf_matrix(df, user_profile)
        df_scored = score_and_rank(df, job_vectors, user_vector)

        # PHASE 3: Top-N Output
        display_recommendations(df_scored, user_profile)

        # Ask to run again
        print(f"\n{'='*55}")
        again = input("  Try with different skills? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Recommendation engine shut down. Good luck! 🚀\n")
            break

if __name__ == "__main__":
    main()