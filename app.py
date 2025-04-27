import streamlit as st
import numpy as np

# Load trained Q-table
q_table = np.load("trained_q_table_balanced.npy")  # Make sure this file exists in the same folder
st.sidebar.caption(f"Q-table shape: {q_table.shape}")  # For debugging

# Define safe discretization function
def categorize_input(value, bins):
    bins = sorted(bins)
    if value < bins[0]:
        return 0
    elif value >= bins[-1]:
        return len(bins)  # Assigns to uppermost bin
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 1

# Define bin thresholds
bin_defs = {
    'Age': [10, 25, 60],
    'BMI': [18, 25, 30],
    'WBCs': [4, 10, 15],
    'Na': [130, 138, 145],
    'Hb': [10, 13, 16],
    'K': [3.5, 4.5, 5.5]
}

# Define actions
actions = ["Open Surgery", "Laparoscopy"]

# Sidebar: patient input
st.sidebar.title("Patient Profile")
gender = st.sidebar.radio("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 1, 100, 25)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 22.0)
wbcs = st.sidebar.slider("WBC Count", 0.0, 30.0, 7.0)
na = st.sidebar.slider("Sodium (Na)", 120.0, 160.0, 138.0)
hb = st.sidebar.slider("Hemoglobin (Hb)", 5.0, 20.0, 13.5)
k = st.sidebar.slider("Potassium (K)", 2.0, 6.5, 4.2)

# Main section
st.title("🩺 Surgical Treatment Recommender")
st.subheader("Based on Q-Learning (Balanced Data)")
st.write("This system uses reinforcement learning to recommend the best treatment option based on patient characteristics.")

# Discretize input to Q-table indices
gender_bin = 1 if gender == "Male" else 0
age_bin = categorize_input(age, bin_defs['Age'])
bmi_bin = categorize_input(bmi, bin_defs['BMI'])
wbcs_bin = categorize_input(wbcs, bin_defs['WBCs'])
na_bin = categorize_input(na, bin_defs['Na'])
hb_bin = categorize_input(hb, bin_defs['Hb'])
k_bin = categorize_input(k, bin_defs['K'])

state = (gender_bin, age_bin, bmi_bin, wbcs_bin, na_bin, hb_bin, k_bin)

# Validate state dimensions
try:
    q_values = q_table[state]
    action = np.argmax(q_values)
    st.success(f"✅ Recommended Treatment: **{actions[action]}**")

    with st.expander("View Q-values for each action"):
        for i, val in enumerate(q_values):
            st.write(f"{actions[i]}: {val:.4f}")

    with st.expander("Discretized State Mapping"):
        st.write(f"Discretized State: {state}")
except IndexError:
    st.error(f"⚠️ Invalid state: {state}. Please adjust input values.")

# --- Sticky Footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #999999;
        text-align: center;
        font-size: 12px;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        © Ireri Mugambi 2025. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
