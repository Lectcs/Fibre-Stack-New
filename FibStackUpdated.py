from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import streamlit as st
import re

st.title('Composite Fiber Stack — Fmax Predictor (Small Dataset)')

# Load small dataset
@st.cache_data
def load_data():
    df = pd.DataFrame({
        'Type': ['G/G/G/G/G/G','K/K/K/K/K/K','C/C/C/C/C/C','[K/G/C/C/G/K]','[C/G/K/K/G/C]','[G/C/K/K/C/G]'],
        'Area_mm2':[66.5,68.4,70.3,68.22,70.12,69.12],
        'Fmax_N':[17852,19225,19020,21852,21200,21560],
        'Tensile_MPa':[268.45,281.07,270.55,320.32,302.34,311.92]
    })

    def extract_stack(type_str):
        items = re.findall(r'[CKG]', str(type_str))
        # Return empty list instead of None if we don't have exactly 6 items
        return items if len(items)==6 else []

    def to_int(f):
        return {'G':0,'C':1,'K':2}[f]

    stacks = df['Type'].apply(extract_stack)
    # Filter rows where stacks is not an empty list
    valid_rows = stacks.apply(len) == 6
    df = df[valid_rows].reset_index(drop=True)
    
    # Now stacks only contains valid entries
    stacks = stacks[valid_rows].reset_index(drop=True)

    # Encode stack layers
    for i in range(6):
        df[f'f{i+1}'] = stacks.apply(lambda x: to_int(x[i]))
    
    X = df[[f'f{i+1}' for i in range(6)] + ['Area_mm2']]
    y = df['Fmax_N']
    return df, X, y

df, X, y = load_data()

# Train Random Forest
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# User selects stack
st.write('Select a 6-layer fiber stack:')
fiber_names = ['G','C','K']
fiber_ints = {'G':0,'C':1,'K':2}
cols = st.columns(6)
selected = []
for i in range(6):
    sel = cols[i].selectbox(f'Layer {i+1}', fiber_names, index=0)
    selected.append(fiber_ints[sel])

# User input for Area
Area_input = st.number_input('Cross-sectional Area (mm²)', min_value=10.0, max_value=200.0, value=70.0)

if st.button('Predict Fmax'):
    X_input = np.array(selected + [Area_input]).reshape(1, -1)
    Fmax_pred = model.predict(X_input)[0]
    Tensile_pred = Fmax_pred / Area_input

    st.subheader('Predicted Properties:')
    st.write(f"**Fmax_N (Maximum Load)**: {Fmax_pred:.2f} N")
    st.write(f"**Calculated Tensile Strength**: {Tensile_pred:.2f} MPa")