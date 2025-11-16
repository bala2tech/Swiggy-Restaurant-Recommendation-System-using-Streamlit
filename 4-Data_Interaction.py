import streamlit as st
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import os

# ===== FULL PAGE MODE =====
st.set_page_config(page_title="Swiggy Recommender", layout="wide")

# ------------ BASE PATH ------------ #
BASE = r"D:\python_programs\Swiggy Recommendation"
path = lambda f: os.path.join(BASE, f)

# ------------ LOAD DATA ------------ #
@st.cache_resource
def load_data():
    cleaned = pd.read_csv(path("cleaned_data.csv"))
    other = pd.read_csv(path("encoded_other_columns.csv"))
    X_ohe = sp.load_npz(path("encoded_features.npz"))
    encoder = pickle.load(open(path("encoder.pkl"), "rb"))

    # clean other_df
    for c in other.columns:
        if other[c].dtype == "object":
            other[c] = other[c].fillna("Unknown")
            try: 
                other[c] = pd.to_numeric(other[c])
            except: 
                other[c] = other[c].astype("category").cat.codes
        else:
            other[c] = other[c].fillna(0)

    X_other = sp.csr_matrix(other.values)
    X = sp.hstack([X_other, X_ohe], format="csr")
    return cleaned, other, X, encoder

cleaned_df, other_df, X, encoder = load_data()

# ------------ NN MODEL ------------ #
@st.cache_resource
def get_nn():
    model = NearestNeighbors(metric="cosine", n_neighbors=50)   # Get more so filtering works
    model.fit(X)
    return model

nn_model = get_nn()

# ------------ RECOMMEND (FIXED RATING FILTER) ------------ #
def recommend(city, cuisine, rating, cost, k):

    # build input vector
    tmp = other_df.iloc[:1].copy()
    tmp["rating"], tmp["cost"] = rating, cost

    for c in tmp.columns:
        if tmp[c].dtype == "object":
            try: 
                tmp[c] = pd.to_numeric(tmp[c])
            except: 
                tmp[c] = tmp[c].astype("category").cat.codes

    X_input = sp.hstack([
        sp.csr_matrix(tmp.values),
        sp.csr_matrix(encoder.transform([[city, cuisine]]))
    ])

    # get 50 nearest neighbors
    dist, idx = nn_model.kneighbors(X_input, n_neighbors=50)

    result = cleaned_df.iloc[idx.flatten()].copy()
    result["similarity"] = 1 - dist.flatten()

    # ⭐ FILTER STRICTLY BY MINIMUM RATING
    result = result[result["rating"] >= rating]

    # sort by similarity
    result = result.sort_values("similarity", ascending=False)

    return result.head(k)

# ------------ VIBRANT UI CSS (FULL WIDTH) ------------ #
st.markdown("""
<style>

html, body, .main { width: 100%; height: 100%; }

/* wider layout */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Larger labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: white;
}

.stSelectbox div, .stSlider div, .stNumberInput input {
    font-size: 1.15rem !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#ff512f,#f09819);
    color:white; border:none; border-radius:14px;
    padding:14px 26px; font-size:1.3rem;
    box-shadow:0 5px 10px rgba(0,0,0,0.35);
    transition:0.25s; width:100%;
}
.stButton>button:hover {
    transform: scale(1.06);
    background: linear-gradient(135deg,#f09819,#ff512f);
}

/******** RESULT CARD — MAGENTA → CYAN ********/
.rec {
    background: linear-gradient(135deg,#ff00c3,#00eaff);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 3px solid white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.45);
    transition: 0.25s ease;
    color: black;
    font-weight: bold;
    font-size: 1.25rem;
}

.rec:hover {
    transform: scale(1.03);
    box-shadow: 0 14px 26px rgba(0,0,0,0.55);
}

.rec h3 {
    font-weight: 900;
    font-size: 1.8rem;
    color: #ffffff;
    text-shadow: 2px 2px 4px #00000070;
}

.rec a {
    background: white;
    padding: 8px 14px;
    border-radius: 10px;
    display: inline-block;
    font-weight: 900;
    font-size: 1.1rem;
    color: black !important;
    text-decoration: none;
}
.rec a:hover {
    background: #ffe6ff;
}

h1 { 
    text-shadow: 2px 2px 6px #0007; 
    font-size: 3rem !important;
}

</style>
""", unsafe_allow_html=True)

# ------------ TITLE ------------ #
st.markdown(
    "<h1 style='text-align:center;color:white;'>🍽️ Swiggy Restaurant Recommender</h1>",
    unsafe_allow_html=True
)

# ------------ LAYOUT ------------ #
left, right = st.columns([1.2, 2.5])

with left:
    st.markdown("## 🎛️ Choose Filters")

    city = st.selectbox("🏙️ City", sorted(cleaned_df["city"].unique()))
    cuisine = st.selectbox("🍽️ Cuisine", sorted(cleaned_df["cuisine"].unique()))
    rating = st.slider("⭐ Minimum Rating", 1.0, 5.0, 4.5)
    cost = st.number_input("💰 Maximum Cost (₹)", 50, 2000, 400)
    k = st.slider("📊 Results to Show", 5, 20, 10)

    run = st.button("🔍 Show Recommendations")

with right:
    if run:
        st.info("⚙️ Fetching best matches…")
        res = recommend(city, cuisine, rating, cost, k)

        st.success(f"🎉 {len(res)} restaurants found!")

        for _, r in res.iterrows():
            st.markdown(f"""
            <div class="rec">
                <h3>🍴 {r['name']}</h3>
                <p>📍 {r['city']} | 🍽️ {r['cuisine']}</p>
                <p>⭐ Rating: {r['rating']} | 💰 Cost: ₹{r['cost']}</p>
                <a href="{r['link']}" target="_blank">🔗 Open in Swiggy</a>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(
            "<p style='color:white;font-size:1.4rem;font-weight:bold;'>👈 Select filters to begin!</p>",
            unsafe_allow_html=True
        )
