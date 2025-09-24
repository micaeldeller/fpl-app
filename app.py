import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

st.set_page_config(layout="wide", page_title="FPL Transfer Planner")

# ---------- Helpers ----------
def normalize_cost(df, cost_col="now_cost"):
    """Convert cost to float (£m) if stored as tenths of millions."""
    if cost_col not in df.columns:
        return df, None
    median = df[cost_col].median()
    if median > 100:  # already in £m
        df["cost_float"] = df[cost_col]
    else:  # stored in tenths
        df["cost_float"] = df[cost_col] / 10.0
    return df, "cost_float"

def position_map(code):
    return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(code, "UNK")

# ---------- Uploads ----------
st.sidebar.title("Upload your data")
stats_file = st.sidebar.file_uploader("Upload player stats CSV", type="csv")
fixtures_file = st.sidebar.file_uploader("Upload fixtures CSV", type="csv")

if not stats_file or not fixtures_file:
    st.warning("Upload both CSVs (stats + fixtures) to continue")
    st.stop()

stats = pd.read_csv(stats_file)
fixtures = pd.read_csv(fixtures_file)

# Detect cost column
stats, cost_col = normalize_cost(stats, "now_cost")
if cost_col is None:
    st.error("No valid cost column found in stats CSV.")
    st.stop()

# Add expected points column
if "points_per_game" in stats.columns:
    stats["expected"] = stats["points_per_game"].astype(float)
elif "total_points" in stats.columns:
    stats["expected"] = stats["total_points"] / 10.0
else:
    stats["expected"] = np.random.rand(len(stats))  # fallback random

# Map element_type → position (GK/DEF/MID/FWD)
if "element_type" in stats.columns:
    stats["position"] = stats["element_type"].apply(position_map)
elif "position" not in stats.columns:
    st.error("No position information available (need element_type or position column).")
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.subheader("Gameweek & Transfers")
gameweek = st.sidebar.number_input("Current Gameweek", min_value=1, max_value=38, value=1, step=1)
max_transfers = st.sidebar.number_input("Max transfers", min_value=0, max_value=15, value=2, step=1)
transfer_budget = st.sidebar.number_input("Money available (£m)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

# ---------- Current team selection ----------
st.subheader("Pick Your Current Team (15 players)")
with st.expander("Select players"):
    available_cols = [c for c in ["web_name", "team", "position", cost_col, "expected"] if c in stats.columns]
    edited = st.data_editor(
        stats[available_cols],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

    column_config={
            "web_name": st.column_config.TextColumn("Player"),
            "team": st.column_config.TextColumn("Team"),
            "position": st.column_config.TextColumn("Pos"),
            cost_col: st.column_config.NumberColumn("Cost (£m)", format="%.1f"),
            "expected": st.column_config.NumberColumn("Exp. Pts", format="%.2f"),
        },
    disabled=[cost_col, "expected", "position", "team"],
    )

# Tick selection → current team
if "selected_rows" in edited:
    st.info("Note: new Streamlit versions allow row selection directly. For now, pick manually.")
current_team = st.multiselect(
    "Choose your current 15 players by name",
    options=stats["web_name"].tolist(),
)

if len(current_team) != 15:
    st.warning("Please select exactly 15 players to continue.")
    st.stop()

current = stats[stats["web_name"].isin(current_team)].copy()
current_cost = current[cost_col].sum()

# ---------- Optimizer ----------
if st.button("Suggest Transfers"):
    players = stats.to_dict("records")
    model = LpProblem(name="fpl_squad", sense=LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(len(players))}

    # Objective: maximize expected points
    model += lpSum([players[i]["expected"] * x[i] for i in x])

    # Constraints
    model += lpSum([players[i][cost_col] * x[i] for i in x]) <= current_cost + transfer_budget
    model += lpSum([x[i] for i in x]) == 15

    # Position constraints (basic FPL rules)
    model += lpSum([x[i] for i in x if players[i].get("position") == "GK"]) == 2
    model += lpSum([x[i] for i in x if players[i].get("position") == "DEF"]) >= 5
    model += lpSum([x[i] for i in x if players[i].get("position") == "MID"]) >= 5
    model += lpSum([x[i] for i in x if players[i].get("position") == "FWD"]) >= 3

    model.solve()
    selected_idx = [i for i in x if x[i].value() == 1.0]
    new_team = pd.DataFrame([players[i] for i in selected_idx])

    # Compare with current team
    transfers_out = current[~current["web_name"].isin(new_team["web_name"])]
    transfers_in = new_team[~new_team["web_name"].isin(current["web_name"])]

    # Transfer count check
    if len(transfers_in) > max_transfers:
        st.warning(
            f"Optimizer suggested {len(transfers_in)} transfers, "
            f"but you only allowed {max_transfers}."
        )

    # ---------- Show results ----------
    st.subheader("Suggested New Team")
    st.dataframe(new_team[["web_name", "team", "position", cost_col, "expected"]])

    st.subheader("Transfers")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Transfers OUT**")
        st.dataframe(transfers_out[["web_name", "position", cost_col]])
    with col2:
        st.write("**Transfers IN**")
        st.dataframe(transfers_in[["web_name", "position", cost_col, "expected"]])

    # ---------- Formation view ----------
    st.subheader("Formation View")
    formation = {
        "GK": new_team[new_team["position"] == "GK"],
        "DEF": new_team[new_team["position"] == "DEF"],
        "MID": new_team[new_team["position"] == "MID"],
        "FWD": new_team[new_team["position"] == "FWD"],
    }

    # pick XI and bench
    bench = pd.concat([
        formation["GK"].iloc[1:2],
        formation["DEF"].iloc[4:5],
        formation["MID"].iloc[4:5],
        formation["FWD"].iloc[2:3],
    ])

    st.markdown("### Starting XI")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        row = " | ".join(formation[pos].head(4)["web_name"].tolist())
        st.write(f"**{pos}:** {row}")

    st.markdown("### Bench")
    st.write(" | ".join(bench["web_name"].tolist()))
