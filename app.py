import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

st.set_page_config(layout="wide", page_title="FPL Helper")

# ---------- Helpers ----------
def normalize_cost(df, cost_col="now_cost"):
    if cost_col not in df.columns:
        return df, None
    median = df[cost_col].median()
    if median > 100:
        df["cost_float"] = df[cost_col]
    else:
        df["cost_float"] = df[cost_col] / 10.0
    return df, "cost_float"

# ---------- Uploads ----------
st.sidebar.title("Upload your data")
stats_file = st.sidebar.file_uploader("Upload player stats CSV", type="csv")
fixtures_file = st.sidebar.file_uploader("Upload fixtures CSV", type="csv")

if not stats_file or not fixtures_file:
    st.warning("Upload both CSV files to continue")
    st.stop()

stats = pd.read_csv(stats_file)
fixtures = pd.read_csv(fixtures_file)

# Detect cost column
stats, cost_col = normalize_cost(stats, "now_cost")
if cost_col is None:
    st.error("No valid cost column found in stats CSV.")
    st.stop()

# Add basic expected points
if "points_per_game" in stats.columns:
    stats["expected"] = stats["points_per_game"].astype(float)
elif "total_points" in stats.columns:
    stats["expected"] = stats["total_points"] / 10.0
else:
    stats["expected"] = np.random.rand(len(stats))

st.sidebar.subheader("Budget & constraints")
budget = st.sidebar.number_input("Budget (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)

# ---------- Optimizer ----------
if st.button("Run optimizer"):
    players = stats.to_dict("records")
    model = LpProblem(name="fpl_squad", sense=LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(len(players))}

    model += lpSum([players[i]["expected"] * x[i] for i in x])  # objective
    model += lpSum([players[i][cost_col] * x[i] for i in x]) <= budget
    model += lpSum([x[i] for i in x]) == 15

    model.solve()
    selected_idx = [i for i in x if x[i].value() == 1.0]
    selected = pd.DataFrame([players[i] for i in selected_idx])

    st.subheader("Selected Squad")
    st.dataframe(selected)
