
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write(
    "Predict student final marks using Machine Learning "
    "and previous academic information."
)

# -------------------------------
# 2. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "data" / "student-mat.csv"
    df = pd.read_csv(file_path, sep=";")
    return df

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# -------------------------------
# 3. Select Features and Target
# -------------------------------
features = ["studytime", "failures", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# -------------------------------
# 4. Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 5. Train ML Models
# -------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
forest_model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Models
# -------------------------------
linear_predictions = linear_model.predict(X_test)
forest_predictions = forest_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
forest_mae = mean_absolute_error(y_test, forest_predictions)

linear_r2 = r2_score(y_test, linear_predictions)
forest_r2 = r2_score(y_test, forest_predictions)

# -------------------------------
# 7. Dashboard Summary
# -------------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Input Features", len(features))
col3.metric("Maximum Grade", "20")

st.divider()

# -------------------------------
# 8. Student Input Form
# -------------------------------
st.subheader("🔮 Predict Student Marks")

with st.form("prediction_form"):
    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "More than 10 hours"
        }[x]
    )

    failures = st.number_input(
        "Previous Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=4
    )

    g1 = st.slider("First Period Grade (G1)", 0, 20, 12)
    g2 = st.slider("Second Period Grade (G2)", 0, 20, 13)

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        ["Random Forest", "Linear Regression"]
    )

    submitted = st.form_submit_button(
        "Predict Final Marks",
        use_container_width=True
    )

if submitted:
    student = pd.DataFrame(
        [[studytime, failures, absences, g1, g2]],
        columns=features
    )

    if model_choice == "Random Forest":
        prediction = forest_model.predict(student)[0]
    else:
        prediction = linear_model.predict(student)[0]

    prediction = max(0, min(20, float(prediction)))

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Predicted Final Grade",
        f"{prediction:.1f} / 20"
    )

    if prediction < 10:
        category = "Needs Improvement"
        st.warning(
            "The predicted grade is below 10. "
            "Additional academic support may help."
        )
    elif prediction < 14:
        category = "Average"
        st.info("Regular revision and practice may help improve performance.")
    elif prediction < 17:
        category = "Good"
        st.success("The predicted performance is good.")
    else:
        category = "Excellent"
        st.success("The predicted performance is excellent.")

    result_col2.metric("Performance Category", category)

    st.caption(
        "This is an estimate, not a guaranteed grade or an official academic decision."
    )

# -------------------------------
# 9. Model Comparison
# -------------------------------
st.divider()
st.subheader("🤖 Machine Learning Model Evaluation")

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, forest_mae],
    "R2 Score": [linear_r2, forest_r2]
})

st.dataframe(
    comparison.round(3),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Lower MAE is better. An R² score closer to 1 generally indicates a better fit."
)

# -------------------------------
# 10. Grade Distribution Chart
# -------------------------------
st.divider()
st.subheader("📈 Student Grade Analysis")

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(df["G3"], bins=15, edgecolor="black")
ax.set_title("Distribution of Final Grades")
ax.set_xlabel("Final Grade (G3)")
ax.set_ylabel("Number of Students")

st.pyplot(fig)
plt.close(fig)

st.subheader("📋 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.caption(
    "Dataset: UCI Student Performance. "
    "The model uses earlier period grades to estimate the final grade."
)
