## app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Alpha Factor Dashboard", layout="wide")
st.title("Alpha Factor Discovery Pipeline")

## ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Long/short quintile size", 5, 30, 10)