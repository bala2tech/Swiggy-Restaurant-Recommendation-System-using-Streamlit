# Swiggy-Restaurant-Recommendation-System-using-Streamlit

Recommendation Systems and Data Analytics

The goal of this project is to develop a data-driven recommendation system that suggests restaurants similar to user preferences. The model uses clustering or similarity-based algorithms (K-Means, Cosine Similarity) to generate recommendations and presents them through a clean and interactive Streamlit app.

🧠 Skills & Technologies

Python

Pandas / NumPy

Scikit-learn

One-Hot Encoding

Clustering / Cosine Similarity

Streamlit (Web App Development)

Data Preprocessing and Cleaning

🔍 Approach
1. Data Understanding & Cleaning

Removed duplicates and handled missing values.

Saved the cleaned dataset as cleaned_data.csv.

2. Data Preprocessing

Applied One-Hot Encoding to categorical features (city, cuisine).

Saved the encoded dataset as encoded_data.csv.

Stored encoder as encoder.pkl.

3. Recommendation Methodology

Used K-Means clustering or Cosine Similarity to identify similar restaurants.

Mapped indices from encoded data to original cleaned data.

4. Streamlit Application

User-friendly interface to input preferences (city, cuisine, rating, cost, etc.).

Displays top restaurant recommendations dynamically.

🧾 Results

✅ Cleaned dataset ready for model training

✅ Encoded dataset with One-Hot Encoding

✅ Working Streamlit app that provides personalized restaurant recommendations

✅ Serialized encoder and reproducible workflow

💼 Business Use Cases

🎯 Personalized Recommendations: Help users find restaurants matching their tastes.

💬 Improved Customer Experience: Simplify restaurant discovery.

📈 Market Insights: Analyze customer preferences and trends.

⚙️ Operational Efficiency: Help businesses tailor offerings to demand.
