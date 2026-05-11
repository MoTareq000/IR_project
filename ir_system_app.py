"""
IR System — Wuzzuf Job Search Engine (Streamlit UI)
Converted from IR_System_v2.ipynb

Architecture:
- TF-IDF search on job title (name column)
- Dropdown hard filters: Department, Experience, Employment Type, Work Mode, Governorate
- Evaluation tab: Precision, Recall, F1 (lenient: >=50% of active filters match)
"""

import re
import math
import warnings
from collections import defaultdict, Counter

import kagglehub
import nltk
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wuzzuf Job Search Engine",
    page_icon="🔍",
    layout="wide",
)

# ── NLTK downloads (cached) ──────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["wordnet", "omw-1.4", "punkt", "averaged_perceptron_tagger_eng", "stopwords"]:
        nltk.download(pkg, quiet=True)

download_nltk()

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords as nltk_stopwords

# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_employment_type(t: str) -> str:
    t = t.lower()
    if "full time"   in t: return "Full Time"
    if "internship"  in t: return "Internship"
    if "part time"   in t: return "Part Time"
    if "freelance"   in t: return "Freelance"
    if "shift based" in t: return "Shift Based"
    return "Other"

def _parse_work_mode(t: str) -> str:
    t = t.lower()
    if "remote" in t:                    return "Remote"
    if "hybrid" in t:                    return "Hybrid"
    if re.search(r"on.?site|onsite", t): return "On-site"
    return "Other"

def _parse_department(cat: str) -> str:
    parts = [p.strip() for p in cat.split("|")]
    for p in parts[2:]:
        if len(p) > 3:
            return p.strip()
    return parts[0].strip() if parts else "Other"

def _parse_experience(cat: str) -> str:
    parts = [p.strip() for p in cat.split("|")]
    if len(parts) > 1:
        lvl = parts[1].strip()
        if lvl in {"Experienced", "Entry Level", "Manager", "Senior Management", "Student"}:
            return lvl
    return "Other"

def _parse_governorate(loc: str) -> str:
    loc = loc.lower()
    if "cairo"       in loc: return "Cairo"
    if "giza"        in loc: return "Giza"
    if "alexandria"  in loc: return "Alexandria"
    if "sharqia"     in loc: return "Sharqia"
    if "monuf"       in loc: return "Monufya"
    if "dakahlia"    in loc or "mansoura" in loc: return "Dakahlia"
    if "gharbia"     in loc or "tanta"    in loc: return "Gharbia"
    if "suez"        in loc: return "Suez"
    if "ismailia"    in loc: return "Ismailia"
    if "south sinai" in loc or "sharm"    in loc: return "South Sinai"
    if "red sea"     in loc or "hurghada" in loc: return "Red Sea"
    if "damietta"    in loc: return "Damietta"
    if "beni suef"   in loc: return "Beni Suef"
    if "saudi"       in loc: return "Saudi Arabia"
    if "dubai"       in loc or "emirates" in loc or "uae" in loc: return "UAE"
    return "Other"

# ── Load & preprocess data ───────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    import os
    local_file = "Cleaned_Wuzzuf_Jobs.csv"
    if os.path.exists(local_file):
        df_raw = pd.read_csv(local_file)
    else:
        path = kagglehub.dataset_download("ahmedhazemelabady/wuzzuf-job-listings-dataset-egypt-january-2025")
        df_raw = pd.read_csv(path + "/Cleaned_Wuzzuf_Jobs.csv")
    df = df_raw.copy()
    df["employment_type"] = df["type"].apply(_parse_employment_type)
    df["work_mode"]       = df["type"].apply(_parse_work_mode)
    df["department"]      = df["job category"].apply(_parse_department)
    df["experience"]      = df["job category"].apply(_parse_experience)
    df["governorate"]     = df["location"].apply(_parse_governorate)
    return df

def _dropdown_options(series: pd.Series, min_count: int = 5) -> list:
    counts = series.value_counts()
    return ["All"] + sorted(counts[counts >= min_count].index.tolist())

# ── Text preprocessing ───────────────────────────────────────────────────────

@st.cache_resource
def get_lemmatizer():
    return WordNetLemmatizer()

@st.cache_resource
def get_stop_words():
    return set(nltk_stopwords.words("english"))

def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess(text: str, lemmatizer, stop_words, keep_tech_symbols: bool = True) -> list:
    text = str(text).lower()
    if keep_tech_symbols:
        text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) >= 2 and not t.isdigit()]
    tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return tokens

@st.cache_data
def tokenize_df(df: pd.DataFrame) -> pd.DataFrame:
    lemmatizer = get_lemmatizer()
    stop_words = get_stop_words()
    df = df.copy()
    df["tokens"] = df["name"].apply(lambda x: preprocess(x, lemmatizer, stop_words))
    return df

# ── Index building ───────────────────────────────────────────────────────────

@st.cache_data
def build_index(df: pd.DataFrame):
    inv_idx   = defaultdict(list)
    doc_store = {}
    doc_freq  = defaultdict(int)
    N         = len(df)

    for doc_id, row in df.iterrows():
        tokens            = row["tokens"]
        doc_store[doc_id] = row["name"]
        term_counts       = Counter(tokens)
        for term, tf in term_counts.items():
            inv_idx[term].append((doc_id, tf))
        for term in term_counts:
            doc_freq[term] += 1

    return dict(inv_idx), doc_store, dict(doc_freq), N

# ── Search ───────────────────────────────────────────────────────────────────

def search(
    df, inv_idx, doc_freq, N,
    query="",
    department="All",
    experience="All",
    employment_type="All",
    work_mode="All",
    governorate="All",
    top_k=20,
    threshold_ratio=0.2,
) -> pd.DataFrame:
    lemmatizer = get_lemmatizer()
    stop_words = get_stop_words()
    scores = defaultdict(float)

    if query.strip():
        query_tokens = preprocess(query, lemmatizer, stop_words)
        for term in query_tokens:
            if term in inv_idx:
                df_term = doc_freq[term]
                idf = math.log((N + 1) / (df_term + 1)) + 1
                for doc_id, tf in inv_idx[term]:
                    scores[doc_id] += tf * idf

        if scores:
            scored_ids = list(scores.keys())
        else:
            scored_ids = []
    else:
        scored_ids = list(df.index)
        scores = defaultdict(float)

    if not scored_ids:
        return df.iloc[0:0]

    result = df.loc[scored_ids].copy()
    result["score"] = result.index.map(lambda i: scores[i])

    if department      != "All": result = result[result["department"]      == department]
    if experience      != "All": result = result[result["experience"]      == experience]
    if employment_type != "All": result = result[result["employment_type"] == employment_type]
    if work_mode       != "All": result = result[result["work_mode"]       == work_mode]
    if governorate     != "All": result = result[result["governorate"]     == governorate]

    result = result.sort_values("score", ascending=False).head(top_k)
    display_cols = ["name", "company", "location", "type", "job category", "score"]
    return result[[c for c in display_cols if c in result.columns]]

# ── Evaluation helpers ───────────────────────────────────────────────────────

def _precision_recall_f1(retrieved_set, relevant_set):
    tp = len(retrieved_set & relevant_set)
    p  = tp / len(retrieved_set) if retrieved_set else 0.0
    r  = tp / len(relevant_set)  if relevant_set  else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4), tp

def get_relevant_docs(
    df,
    query="",
    department="All",
    experience="All",
    employment_type="All",
    work_mode="All",
    governorate="All",
    min_match_fraction=0.5
):

    filters = {
        "department": department,
        "experience": experience,
        "employment_type": employment_type,
        "work_mode": work_mode,
        "governorate": governorate,
    }

    active = {col: val for col, val in filters.items() if val != "All"}

    # start with all docs
    relevant_mask = pd.Series(True, index=df.index)

    # apply dropdown filters
    for col, val in active.items():
        relevant_mask &= (df[col] == val)

    # apply query relevance
    if query.strip():

        lemmatizer = get_lemmatizer()
        stop_words = get_stop_words()

        query_tokens = set(preprocess(query, lemmatizer, stop_words))

        def has_query_match(tokens):
            return len(query_tokens.intersection(set(tokens))) > 0

        relevant_mask &= df["tokens"].apply(has_query_match)

    return set(df[relevant_mask].index.tolist())

def evaluate_query(df, inv_idx, doc_freq, N,
                   query="", department="All", experience="All",
                   employment_type="All", work_mode="All", governorate="All",
                   k=20, threshold_ratio=0.2, min_match_fraction=0.5) -> dict:
    relevant = get_relevant_docs(
    df,
    query=query,
    department=department,
    experience=experience,
    employment_type=employment_type,
    work_mode=work_mode,
    governorate=governorate,
    min_match_fraction=min_match_fraction,
)
    results_df = search(
        df, inv_idx, doc_freq, N,
        query=query, department=department, experience=experience,
        employment_type=employment_type, work_mode=work_mode,
        governorate=governorate, top_k=k, threshold_ratio=threshold_ratio
    )

    retrieved_ids = results_df.index.tolist()
    retrieved_set = set(retrieved_ids)

    p, r, f1, tp = _precision_recall_f1(retrieved_set, relevant)

    return {
        "query": query,
        "department": department,
        "experience": experience,
        "employment_type": employment_type,
        "work_mode": work_mode,
        "governorate": governorate,
        "k": k,
        "retrieved": len(retrieved_set),
        "relevant_pool": len(relevant),
        "tp": tp,
        "Precision": p,
        "Recall": r,
        "F1": f1,
    }

# ── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.title("Wuzzuf Job Search Engine")
    st.caption("TF-IDF title search · structured dropdown filters · built-in evaluation")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")
        st.caption("Using KaggleHub dataset (no upload needed)")

    # ── Build pipeline ──────────────────────────────────────────────────────
    with st.spinner("Preprocessing & building index..."):
        df = load_data()
        df = tokenize_df(df)
        inv_idx, doc_store, doc_freq, N = build_index(df)

        st.sidebar.success(f"Loaded {len(df):,} jobs")

        DEPT_OPTIONS = _dropdown_options(df["department"])
        EXP_OPTIONS  = _dropdown_options(df["experience"])
        EMP_OPTIONS  = _dropdown_options(df["employment_type"])
        MODE_OPTIONS = _dropdown_options(df["work_mode"])
        GOV_OPTIONS  = _dropdown_options(df["governorate"])

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_search, tab_batch = st.tabs(["Search", "Batch Evaluation"])

    # ── TAB 1: Search ───────────────────────────────────────────────────────
    with tab_search:
        st.subheader("Search Jobs")

        col_q, col_k = st.columns([4, 1])

        with col_q:
            query = st.text_input(
                "Job title keywords",
                placeholder="e.g. data analyst, python developer"
            )

        with col_k:
            top_k = st.number_input(
                "Top K",
                min_value=5,
                max_value=1000,
                value=20,
                step=5
            )

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)

        with fc1:
            department = st.selectbox("Department", DEPT_OPTIONS)

        with fc2:
            experience = st.selectbox("Experience", EXP_OPTIONS)

        with fc3:
            employment_type = st.selectbox("Employment Type", EMP_OPTIONS)

        with fc4:
            work_mode = st.selectbox("Work Mode", MODE_OPTIONS)

        with fc5:
            governorate = st.selectbox("Governorate", GOV_OPTIONS)

        if "last_search_params" not in st.session_state:
            st.session_state.last_search_params = None

        if "eval_requested" not in st.session_state:
            st.session_state.eval_requested = False

        if st.button("Search", type="primary", use_container_width=True, key="search_btn"):
            st.session_state.last_search_params = {
                "query": query,
                "department": department,
                "experience": experience,
                "employment_type": employment_type,
                "work_mode": work_mode,
                "governorate": governorate,
                "top_k": int(top_k),
            }
            st.session_state.eval_requested = False

        params = st.session_state.last_search_params

        if params is not None:
            results = search(
                df, inv_idx, doc_freq, N,
                query=params["query"],
                department=params["department"],
                experience=params["experience"],
                employment_type=params["employment_type"],
                work_mode=params["work_mode"],
                governorate=params["governorate"],
                top_k=params["top_k"],
            )

            st.markdown(f"**{len(results)} result(s)**")

            if results.empty:
                st.warning("No jobs matched your search. Try broadening your filters.")
            else:
                show = results.copy()
                show["score"] = show["score"].round(3)
                show.columns = [c.title() for c in show.columns]

                st.dataframe(show, use_container_width=True, hide_index=True)

                if st.button("Evaluate Quality", use_container_width=True, key="eval_quality_btn"):
                    st.session_state.eval_requested = True

                if st.session_state.eval_requested:
                    r = evaluate_query(
                        df, inv_idx, doc_freq, N,
                        query=params["query"],
                        department=params["department"],
                        experience=params["experience"],
                        employment_type=params["employment_type"],
                        work_mode=params["work_mode"],
                        governorate=params["governorate"],
                        k=params["top_k"],
                    )

                    st.markdown("#### Quality Metrics")

                    m1, m2, m3, m4, m5 = st.columns(5)

                    m1.metric("Precision", r["Precision"])
                    m2.metric("Recall", r["Recall"])
                    m3.metric("F1", r["F1"])
                    m4.metric("Retrieved", r["retrieved"])
                    m5.metric("Relevant Pool", r["relevant_pool"])

                    st.markdown("#### Retrieved Results with Relevance")

                    relevant = get_relevant_docs(
                      df,
                      query=params["query"],
                      department=params["department"],
                      experience=params["experience"],
                      employment_type=params["employment_type"],
                      work_mode=params["work_mode"],
                      governorate=params["governorate"],
                    )

                    rows = []

                    for doc_id, row in results.iterrows():
                        label = "Relevant" if doc_id in relevant else "Not relevant"

                        rows.append({
                            "Relevance": label,
                            "Title": row["name"],
                            "Company": row["company"],
                            "Location": row["location"],
                            "Score": round(row["score"], 3),
                        })

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True
                    )

    # ── TAB 2: Batch Evaluation ─────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Evaluation")

        st.caption(
            "Runs the default suite of 12 test cases across K = 10, 20, 50."
        )

        DEFAULT_TEST_CASES = [
            {"query": "data analyst"},
            {"query": "software engineer"},
            {"query": "graphic designer"},
            {"query": "", "employment_type": "Internship"},
            {"query": "", "work_mode": "Remote"},
            {"query": "", "governorate": "Alexandria"},
            {"query": "data analyst", "employment_type": "Full Time", "governorate": "Cairo"},
            {"query": "python developer", "work_mode": "Remote"},
            {"query": "marketing specialist", "employment_type": "Full Time", "governorate": "Giza"},
            {"query": "hr specialist", "experience": "Experienced", "governorate": "Cairo"},
            {"query": "sales engineer", "employment_type": "Full Time", "work_mode": "On-site"},
            {"query": "customer service", "experience": "Entry Level", "governorate": "Cairo"},
        ]

        k_values = st.multiselect(
            "K values",
            [5, 10, 20, 50, 100],
            default=[10, 20, 50]
        )

        if st.button("Run Batch Evaluation", type="primary", use_container_width=True):

            if not k_values:
                st.warning("Select at least one K value.")

            else:
                rows = []

                progress = st.progress(0)

                total = len(DEFAULT_TEST_CASES) * len(k_values)
                step = 0

                for tc in DEFAULT_TEST_CASES:
                    for k in sorted(k_values):

                        r = evaluate_query(
                            df, inv_idx, doc_freq, N,
                            query=tc.get("query", ""),
                            department=tc.get("department", "All"),
                            experience=tc.get("experience", "All"),
                            employment_type=tc.get("employment_type", "All"),
                            work_mode=tc.get("work_mode", "All"),
                            governorate=tc.get("governorate", "All"),
                            k=k,
                        )

                        rows.append({
                            "Query": r["query"] or "(no text)",
                            "Emp. Type": r["employment_type"],
                            "Work Mode": r["work_mode"],
                            "Governorate": r["governorate"],
                            "K": r["k"],
                            "Retrieved": r["retrieved"],
                            "Relevant Pool": r["relevant_pool"],
                            "Precision": r["Precision"],
                            "Recall": r["Recall"],
                            "F1": r["F1"],
                        })

                        step += 1
                        progress.progress(step / total)

                progress.empty()

                summary = pd.DataFrame(rows)

                st.markdown("#### Mean metrics by K")

                metric_cols = ["Precision", "Recall", "F1"]

                st.dataframe(
                    summary.groupby("K")[metric_cols].mean().round(4),
                    use_container_width=True,
                )

                st.markdown("#### All results")

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

                csv = summary.to_csv(index=False).encode()

                st.download_button(
                    "Download CSV",
                    csv,
                    "batch_eval_results.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()