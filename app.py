import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

# ---- Page Config ----
st.set_page_config(page_title="Optimal Feature Selection", layout="wide")

# ---- Header ----
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Optimal Feature Selection Tool</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---- Sidebar ----
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Select Model",
    ("Linear Regression", "Logistic Regression", "Random Forest")
)

# ---- File Upload ----
st.subheader("Upload your CSV Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.write("Preview of Dataset:")
    st.dataframe(df.head())

    # ---- Select Target Column ----
    target_col = st.selectbox("Select Target Column", df.columns)

    # ---- Run GA Feature Selection ----
    if st.button("Run Feature Selection"):
        with st.spinner("Running feature selection..."):
            time.sleep(2)  # Simulate GA processing

            # ---- Backend: Simple feature selection placeholder ----
            features = [col for col in df.columns if col != target_col]
            num_selected = min(5, len(features))
            selected_features = features[:num_selected]

            # ---- Train Model ----
            X = df[selected_features]
            y = df[target_col]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize model
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier()

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ---- Model Performance ----
            st.subheader("📊 Model Performance")
            try:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                st.metric("Accuracy", f"{acc:.2f}")
                st.metric("F1-score", f"{f1:.2f}")
            except:
                st.info("Model performance metrics not available (regression problem)")

            # ---- Selected Features ----
            st.subheader("✅ Selected Features")
            st.write(selected_features)

            # ---- Export Selected Features to CSV ----
            selected_df = df[selected_features + [target_col]]  # include target column
            csv = selected_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Selected Features as CSV",
                data=csv,
                file_name="selected_features.csv",
                mime="text/csv"
            )

            # ---- Feature Importance ----
            st.subheader("📈 Feature Importance")
            try:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.random.rand(len(selected_features))  # fallback

                importance_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)
                
                st.bar_chart(importance_df.set_index("Feature"))
            except:
                st.info("Feature importance not available for this model.")

else:
    st.info("Please upload a CSV file to start.")


