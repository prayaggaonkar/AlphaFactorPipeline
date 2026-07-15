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

## ── Load results (run main.py first to generate these files) ─────
try:
    cum_ret = pd.read_parquet("backtest/cum_returns.parquet")
    net_pnl = pd.read_parquet("backtest/net_pnl.parquet")

    ## Metrics row
    col1, col2, col3, col4 = st.columns(4)
    ann_ret = round(net_pnl.mean() * 252 * 100, 2)
    sharpe  = round(net_pnl.mean() / net_pnl.std() * (252**0.5), 2)
    max_dd  = round((cum_ret / cum_ret.cummax() - 1).min() * 100, 2)
    win_r   = round((net_pnl > 0).mean() * 100, 1)

    col1.metric("Ann. Return",   f"{ann_ret}%")
    col2.metric("Sharpe Ratio",  sharpe)
    col3.metric("Max Drawdown",  f"{max_dd}%")
    col4.metric("Win Rate",      f"{win_r}%")

    ## Cumulative return chart
    st.subheader("Cumulative returns")
    fig = px.line(cum_ret, labels={"value": "NAV", "index": "Date"})
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.warning("Run main.py first to generate results, then refresh this page.")