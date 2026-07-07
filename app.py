"""HR Employee Attrition Analysis and Prediction — Streamlit app."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_generator import (
    DATA_VERSION,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    MARITAL_STATUS,
    BUSINESS_TRAVEL,
    load_or_generate_data,
)
from model import train_model, predict_employee

st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 0; opacity: 0.9; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading employee dataset...")
def get_data(_version: str = DATA_VERSION) -> pd.DataFrame:
    return load_or_generate_data()


@st.cache_resource(show_spinner="Training prediction model...")
def get_model_artifacts(_version: str = DATA_VERSION):
    df = get_data()
    return train_model(df)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


def tab_overview(df: pd.DataFrame) -> None:
    st.header("Dataset Overview")
    attrition_rate = df["Attrition"].eq("Yes").mean()
    avg_income = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()
    overtime_pct = df["OverTime"].eq("Yes").mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Total Employees", f"{len(df):,}")
    with c2:
        render_metric_card("Attrition Rate", f"{attrition_rate:.1%}")
    with c3:
        render_metric_card("Avg Monthly Income", f"${avg_income:,.0f}")
    with c4:
        render_metric_card("Avg Tenure (yrs)", f"{avg_tenure:.1f}")
    with c5:
        render_metric_card("Overtime Rate", f"{overtime_pct:.1%}")

    st.subheader("Dataset Snapshot")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Shape**")
        st.write(f"Rows: **{df.shape[0]:,}**")
        st.write(f"Columns: **{df.shape[1]}**")
        st.markdown("**Column Types**")
        st.write(f"Numeric: **{df.select_dtypes('number').shape[1]}**")
        st.write(f"Categorical: **{df.select_dtypes('object').shape[1]}**")
        st.markdown("**Missing Values**")
        missing = df.isnull().sum().sum()
        st.write(f"Total missing: **{missing}**")


def tab_eda(df: pd.DataFrame) -> None:
    st.header("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Attrition Rate by Department")
        dept_rate = (
            df.groupby("Department")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
            .sort_values("AttritionRate", ascending=True)
        )
        fig = px.bar(
            dept_rate,
            x="AttritionRate",
            y="Department",
            orientation="h",
            color="AttritionRate",
            color_continuous_scale="RdYlGn_r",
            labels={"AttritionRate": "Attrition Rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Attrition Rate by Overtime")
        ot_rate = (
            df.groupby("OverTime")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig = px.bar(
            ot_rate,
            x="OverTime",
            y="AttritionRate",
            color="OverTime",
            color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
            labels={"AttritionRate": "Attrition Rate (%)"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Attrition Rate by Job Role")
        role_rate = (
            df.groupby("JobRole")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
            .sort_values("AttritionRate", ascending=True)
        )
        fig = px.bar(
            role_rate,
            x="AttritionRate",
            y="JobRole",
            orientation="h",
            color="AttritionRate",
            color_continuous_scale="RdYlGn_r",
            labels={"AttritionRate": "Attrition Rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Attrition by Job Satisfaction")
        sat_rate = (
            df.groupby("JobSatisfaction")["Attrition"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="AttritionRate")
        )
        fig = px.bar(
            sat_rate,
            x="JobSatisfaction",
            y="AttritionRate",
            color="AttritionRate",
            color_continuous_scale="RdYlGn_r",
            labels={"AttritionRate": "Attrition Rate (%)", "JobSatisfaction": "Satisfaction (1-4)"},
        )
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_cols = df.select_dtypes("number").columns.tolist()
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Income Distribution by Attrition Status")
    fig = px.histogram(
        df,
        x="MonthlyIncome",
        color="Attrition",
        barmode="overlay",
        opacity=0.7,
        nbins=40,
        color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
        labels={"MonthlyIncome": "Monthly Income ($)"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def tab_prediction(artifacts: dict) -> None:
    st.header("Attrition Prediction Model")
    st.markdown(
        "A **Random Forest Classifier** tuned via GridSearch is trained on 75% of the dataset, "
        "with threshold optimization on a validation split, and evaluated on the held-out 25% test set."
    )

    metrics = artifacts["metrics"]
    baseline = artifacts["baseline_metrics"]
    threshold = artifacts["threshold"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.1%}",
        delta=f"{metrics['accuracy'] - baseline['accuracy']:+.1%}",
    )
    m2.metric(
        "Precision",
        f"{metrics['precision']:.1%}",
        delta=f"{metrics['precision'] - baseline['precision']:+.1%}",
    )
    m3.metric(
        "Recall",
        f"{metrics['recall']:.1%}",
        delta=f"{metrics['recall'] - baseline['recall']:+.1%}",
    )
    m4.metric(
        "F1 Score",
        f"{metrics['f1']:.1%}",
        delta=f"{metrics['f1'] - baseline['f1']:+.1%}",
    )

    st.subheader("Before vs After Comparison")
    comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Before": [
                f"{baseline['accuracy']:.1%}",
                f"{baseline['precision']:.1%}",
                f"{baseline['recall']:.1%}",
                f"{baseline['f1']:.1%}",
            ],
            "After": [
                f"{metrics['accuracy']:.1%}",
                f"{metrics['precision']:.1%}",
                f"{metrics['recall']:.1%}",
                f"{metrics['f1']:.1%}",
            ],
            "Change": [
                f"{metrics['accuracy'] - baseline['accuracy']:+.1%}",
                f"{metrics['precision'] - baseline['precision']:+.1%}",
                f"{metrics['recall'] - baseline['recall']:+.1%}",
                f"{metrics['f1'] - baseline['f1']:+.1%}",
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.caption(
        f"**Tuned threshold:** {threshold:.2f} (was {baseline['threshold']:.2f}) · "
        f"**Best params:** {artifacts['best_params']}"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = artifacts["confusion_matrix"]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top Feature Importances")
        imp_df = artifacts["feature_importance"].head(15)
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_predict_employee(artifacts: dict) -> None:
    st.header("Predict Attrition for an Employee")
    st.markdown("Enter employee details below to get a live attrition risk prediction.")

    pipeline = artifacts["pipeline"]
    threshold = artifacts["threshold"]

    with st.form("employee_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            department = st.selectbox("Department", list(DEPARTMENTS.keys()))
            job_roles = DEPARTMENTS[department]
            job_role = st.selectbox("Job Role", job_roles)
            monthly_income = st.number_input(
                "Monthly Income ($)", min_value=1500, max_value=20000, value=5000, step=100
            )
            distance = st.number_input(
                "Distance From Home (miles)", min_value=1, max_value=30, value=10
            )

        with c2:
            years_at_company = st.number_input(
                "Years at Company", min_value=0, max_value=40, value=5
            )
            years_in_role = st.number_input(
                "Years in Current Role",
                min_value=0,
                max_value=years_at_company,
                value=min(3, years_at_company),
            )
            total_working_years = st.number_input(
                "Total Working Years",
                min_value=years_at_company,
                max_value=50,
                value=max(years_at_company + 2, 8),
            )
            num_companies = st.number_input(
                "Number of Companies Worked", min_value=1, max_value=10, value=2
            )
            performance = st.selectbox("Performance Rating", [3, 4], index=0)

        with c3:
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            overtime = st.selectbox("OverTime", ["No", "Yes"])
            education = st.selectbox("Education", EDUCATION_LEVELS, index=2)
            marital = st.selectbox("Marital Status", MARITAL_STATUS)
            travel = st.selectbox("Business Travel", BUSINESS_TRAVEL)

        submitted = st.form_submit_button("Predict Attrition Risk", type="primary")

    if submitted:
        employee = {
            "Age": age,
            "Department": department,
            "JobRole": job_role,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "DistanceFromHome": distance,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": overtime,
            "NumCompaniesWorked": num_companies,
            "TotalWorkingYears": total_working_years,
            "PerformanceRating": performance,
            "Education": education,
            "MaritalStatus": marital,
            "BusinessTravel": travel,
        }

        label, prob = predict_employee(pipeline, employee, threshold=threshold)
        risk_pct = prob * 100

        st.divider()
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if label == "Yes":
                st.error(f"**High Attrition Risk** — Predicted: **{label}**")
            else:
                st.success(f"**Low Attrition Risk** — Predicted: **{label}**")

            st.progress(min(prob, 1.0))
            st.markdown(f"### Attrition Probability: **{risk_pct:.1f}%**")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={"suffix": "%"},
                    title={"text": "Attrition Risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if label == "Yes" else "#2ecc71"},
                        "steps": [
                            {"range": [0, 30], "color": "#d5f5e3"},
                            {"range": [30, 60], "color": "#fdebd0"},
                            {"range": [60, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.8,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)


def main() -> None:
    df = get_data()
    artifacts = get_model_artifacts()

    with st.sidebar:
        st.title("HR Attrition Analytics")
        st.markdown("---")
        st.markdown("**Navigation**")
        st.markdown(
            "- **Overview** — dataset summary & key metrics  \n"
            "- **EDA** — interactive exploratory charts  \n"
            "- **Prediction Model** — Random Forest evaluation  \n"
            "- **Predict for an Employee** — live risk scoring"
        )
        st.markdown("---")
        st.caption(
            "Synthetic HR dataset with ~1,200 employee records. "
            "Attrition is modeled with realistic correlations across "
            "satisfaction, overtime, income, and tenure."
        )

    st.title("HR Employee Attrition Analysis & Prediction")
    st.markdown(
        "This dashboard explores employee attrition patterns and trains a "
        "**Random Forest** classifier to predict whether an employee is likely to leave. "
        "Use the tabs below to explore overview metrics, exploratory charts, "
        "model evaluation, and individual risk scoring."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "EDA", "Prediction Model", "Predict for an Employee"]
    )
    with tab1:
        tab_overview(df)
    with tab2:
        tab_eda(df)
    with tab3:
        tab_prediction(artifacts)
    with tab4:
        tab_predict_employee(artifacts)


if __name__ == "__main__":
    main()