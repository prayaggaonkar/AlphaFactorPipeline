import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Alpha Factor Discovery",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #cbd5e1; }
.block-container { padding-top: 2.2rem !important; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #0d1f3c 0%, #080f20 60%, #050912 100%);
    border: 1px solid #162035;
    border-radius: 20px;
    padding: 60px 52px 52px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 48px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 20px;
}
.hero-title em { color: #38bdf8; font-style: normal; }
.hero-body {
    font-size: 17px;
    color: #64748b;
    line-height: 1.8;
    max-width: 700px;
}

.hero-more {
    margin-top: 18px;
    position: relative;
    z-index: 1;
}
.hero-more summary {
    list-style: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.06em;
    color: #38bdf8;
    text-transform: uppercase;
    padding: 10px 16px;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    background: rgba(56,189,248,0.04);
    transition: background 0.15s ease, border-color 0.15s ease;
    width: fit-content;
    outline: none;
}
.hero-more summary:hover {
    background: rgba(56,189,248,0.09);
    border-color: #38bdf8;
}
.hero-more summary::-webkit-details-marker { display: none; }
.hero-more summary::after {
    content: '+';
    font-size: 15px;
    line-height: 1;
    margin-left: 2px;
    transition: transform 0.15s ease;
}
.hero-more[open] summary::after { content: '−'; }
.hero-more-content {
    margin-top: 16px;
    font-size: 15px;
    color: #64748b;
    line-height: 1.8;
    padding: 24px 32px;
    background: rgba(5,10,20,0.4);
    border: 1px solid #131e30;
    border-radius: 12px;
    width: 100%;
    box-sizing: border-box;
}
.hero-more-content strong { color: #e2e8f0; }

.hero-github-wrap {
    position: absolute;
    top: 32px;
    right: 36px;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 18px;
}
.hero-github {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    background: rgba(56,189,248,0.04);
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-decoration: none;
    transition: background 0.15s ease, border-color 0.15s ease;
}
.hero-github:hover { background: rgba(56,189,248,0.09); border-color: #38bdf8; }
.hero-github svg { width: 15px; height: 15px; fill: #38bdf8; }
.hero-author {
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #e2e8f0;
    padding-right: 2px;
}

.info-details { margin: 16px 0; }
.info-details summary {
    list-style: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.06em;
    color: #38bdf8;
    text-transform: uppercase;
    padding: 10px 16px;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    background: rgba(56,189,248,0.04);
    transition: background 0.15s ease, border-color 0.15s ease;
    width: fit-content;
    outline: none;
}
.info-details summary:hover {
    background: rgba(56,189,248,0.09);
    border-color: #38bdf8;
}
.info-details summary::-webkit-details-marker { display: none; }
.info-details summary::marker { content: ""; display: none; }
.hero-more summary::-webkit-details-marker { display: none; }
.hero-more summary::marker { content: ""; display: none; }
.info-details summary::after {
    content: '+';
    font-size: 15px;
    line-height: 1;
    margin-left: 2px;
}
.info-details[open] summary::after { content: '−'; }
.info-details .exp { margin-top: 12px; margin-bottom: 0; }

.sec-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin: 52px 0 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid #111827;
}
.sec-title { font-size: 20px; font-weight: 600; color: #e2e8f0; }
.sec-sub   { font-size: 13px; color: #4a5872; }

.mcard {
    background: #0b1120;
    border: 1px solid #131e30;
    border-radius: 14px;
    padding: 22px 26px 20px;
    height: 100%;
    min-height: 190px;
    display: flex;
    flex-direction: column;
}
.mcard-label {
    font-size: 13.5px;
    font-weight: 600;
    color: #4a5872;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}
.mcard-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 38px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 3px;
}
.mcard-sub { font-size: 12px; color: #4a5872; line-height: 1.55; margin-top: auto; }
.mcard-formula {
    font-size: 12px;
    color: #4a5872;
    margin-top: 4px;
}
.pos { color: #34d399; }
.neg { color: #f87171; }
.neu { color: #38bdf8; }

.exp {
    background: #0b1120;
    border: 1px solid #131e30;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 16px 0;
    font-size: 14px;
    color: #4a5872;
    line-height: 1.8;
}
.exp strong { color: #e2e8f0; }

/* Flow chart */
.flowchart {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 28px 0;
    flex-wrap: nowrap;
}
.flow-box {
    background: #0b1120;
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    min-width: 140px;
    flex: 1;
    max-width: 180px;
    position: relative;
    box-shadow: 0 0 20px rgba(56,189,248,0.08);
}
.flow-box-active {
    border-color: #38bdf8;
    box-shadow: 0 0 20px rgba(56,189,248,0.08);
}
.flow-icon { font-size: 24px; margin-bottom: 10px; }
.flow-label {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 6px;
    line-height: 1.3;
}
.flow-sub {
    font-size: 11px;
    color: #4a5872;
    line-height: 1.5;
}
.flow-arrow {
    display: flex;
    align-items: center;
    padding: 0 6px;
    flex-shrink: 0;
}
.flow-arrow-inner {
    display: flex;
    align-items: center;
}
.flow-arrow-line {
    width: 28px;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, #38bdf8);
    flex-shrink: 0;
}
.flow-arrow-head {
    width: 0;
    height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 7px solid #38bdf8;
    flex-shrink: 0;
}

.ftrow {
    display: flex;
    align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid #0f1724;
    gap: 16px;
}
.ftrow:last-child { border-bottom: none; }
.ft-name  { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7dd3fc; min-width: 110px; }
.ft-desc  { font-size: 12px; color: #4a5872; flex: 1; }
.ft-bar-bg { width: 100px; height: 5px; background: #111827; border-radius: 3px; flex-shrink: 0; }
.ft-bar    { height: 100%; border-radius: 3px; }
.ft-ic     { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; min-width: 64px; text-align: right; }
.ft-badge  { font-size: 10px; padding: 2px 7px; border-radius: 4px; min-width: 52px; text-align: center; }

.stattbl { background: #0b1120; border: 1px solid #131e30; border-radius: 12px; overflow: hidden; }
.strow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid #0f1724;
    font-size: 13px;
}
.strow:last-child { border-bottom: none; }
.strow-label { color: #4a5872; }
.strow-val   { font-family: 'JetBrains Mono', monospace; font-weight: 500; font-size: 13px; }

.codebox {
    background: #060a10;
    border: 1px solid #131e30;
    border-radius: 10px;
    padding: 18px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a5872;
    line-height: 2;
}
.codebox .key { color: #38bdf8; }
.codebox .val { color: #a78bfa; }

.footer {
    text-align: center;
    padding: 48px 0 24px;
    font-size: 12px;
    color: #4a5872;
    border-top: 1px solid #0f1724;
    margin-top: 72px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    try:
        cum = pd.read_parquet("backtest/cum_returns.parquet")["cum_returns"]
        pnl = pd.read_parquet("backtest/net_pnl.parquet")["net_pnl"]
        return cum, pnl, True
    except Exception:
        return None, None, False

cum_returns, net_pnl, data_loaded = load_results()

def mcard(label, val, sub, cls="neu", formula=None):
    formula_html = f'<div class="mcard-formula">{formula}</div>' if formula else ""
    return f"""<div class="mcard">
        <div class="mcard-label">{label}</div>
        <div class="mcard-val {cls}">{val}</div>
        <div class="mcard-sub">{sub}{formula_html}</div>
    </div>"""

def stat_table(rows):
    html = '<div class="stattbl">'
    for label, val, color in rows:
        html += f'<div class="strow"><span class="strow-label">{label}</span><span class="strow-val" style="color:{color};">{val}</span></div>'
    html += '</div>'
    return html


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-github-wrap">
        <a class="hero-github" href="https://github.com/prayaggaonkar/AlphaFactorPipeline" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
            View on GitHub
        </a>
        <div class="hero-author">Prayag Gaonkar</div>
    </div>
    <div class="hero-eyebrow">Quantitative Research · Machine Learning · Stock Markets</div>
    <div class="hero-title">Alpha Factor<br><em>Discovery</em></div>
    <div class="hero-body">
        An end-to-end machine learning pipeline that learns which stock characteristics best predict future returns, ultimately constructing a robust trading strategy. 
        Built with walk-forward validation, SHAP explainability, and realistic transaction cost modeling. 
    </div>
    <details class="hero-more">
        <summary>More about this project</summary>
        <div class="hero-more-content">
            <strong>Overview:</strong> The stock market is notoriously random. Every day, some stocks skyrocket while others plummet.
            This project is a machine learning pipeline that extracts patterns in historical stock data (ie.
            whether a stock is trading below its recent average, or whether it's been unusually volatile)
            to turn randomness into predictability.
            <br><br>
            <strong>How It Earns Money:</strong> Each month, the system buys stocks that the model ranks highest
            while simultaneously betting against the stocks it ranks lowest. This structure means the strategy can lead to earnings 
            regardless of whether the overall market goes up or down.
            <br><br>
            <strong>Technical Details:</strong> The LightGBM model is trained with factors including 
            momentum, mean reversion, and volatility. The gradient boosting nature of the model enables large amounts of data
            processing which reveal subtle patterns. Model predictions are continuously evaluated through a walk-forward framework, where the system is 
            retrained on new data and tested on unseen periods. The resulting signals are converted into a portfolio and analyzed through rigorous backtesting using 
            metrics such as Information Coefficient (IC), Sharpe ratio, annualized return, maximum drawdown, and transaction costs.
        </div>
    </details>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("Run `python main.py` first to generate results, then refresh this page.")
    st.stop()

# ── Compute metrics ───────────────────────────────────────────────────────────
active   = net_pnl[net_pnl != 0]
ann_ret  = active.mean() * 252 * 100
ann_vol  = active.std() * np.sqrt(252) * 100
sharpe   = (active.mean() / active.std() * np.sqrt(252))
max_dd   = ((cum_returns / cum_returns.cummax()) - 1).min() * 100
win_rate = (active > 0).mean() * 100
calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0


# ── KEY RESULTS ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head" style="margin-top:16px;">
    <div class="sec-title">Key Results</div>
    <div class="sec-sub">All metrics are out-of-sample · After transaction costs</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
ret_cls = "pos" if ann_ret > 0 else "neg"
shr_cls = "pos" if sharpe > 0.3 else ("neg" if sharpe < 0 else "neu")
dd_cls  = "neg" if max_dd < -8 else "neu"

with c1: st.markdown(mcard("Annual Return", f"{ann_ret:+.1f}%", "Annualized return on active trading days, after all costs.", ret_cls), unsafe_allow_html=True)
with c2: st.markdown(mcard("Sharpe Ratio", f"{sharpe:.2f}", "Return per unit of risk, annualized. Sharpe = (R̄ − Rf) / σ", shr_cls), unsafe_allow_html=True)
with c3: st.markdown(mcard("Model Information Coefficient", "0.034", "How well the model ranks stocks. >0.02 is industry standard.", "pos"), unsafe_allow_html=True)
with c4: st.markdown(mcard("ICIR", "0.83", "Consistency of predictions across market conditions. >0.5 is good.", "pos"), unsafe_allow_html=True)
with c5: st.markdown(mcard("Max Drawdown", f"{max_dd:.1f}%", "Worst peak-to-trough loss during the backtest.", dd_cls), unsafe_allow_html=True)
with c6: st.markdown(mcard("Win Rate", f"{win_rate:.1f}%", "Percentage of active trading days with positive P&L.", "neu"), unsafe_allow_html=True)

st.markdown("""
<details class="info-details" style="margin-top:20px;">
    <summary>What IC means</summary>
    <div class="exp">
        IC (Information Coefficient) measures how well the model's stock rankings match actual future returns.
        An IC of 0.034 means predictions are 3.4% correlated with real outcomes — small but consistent.
        Professionally, IC > 0.02 is considered genuine, exploitable alpha.
        An ICIR of 0.83 confirms the signal holds across different market periods, not just one lucky stretch.
    </div>
</details>
""", unsafe_allow_html=True)


# ── CUMULATIVE RETURNS ────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Cumulative Returns</div>
    <div class="sec-sub">Starting NAV = $1.00 · After transaction costs</div>
</div>
""", unsafe_allow_html=True)

fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(
    x=cum_returns.index, y=cum_returns.values,
    mode="lines", name="Strategy NAV",
    line=dict(color="#38bdf8", width=2),
    fill="tozeroy", fillcolor="rgba(56,189,248,0.04)",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>NAV: $%{y:.4f}<extra></extra>"
))
fig_nav.add_hline(y=1.0, line_dash="dot", line_color="#1e2535", line_width=1)

# Center the y-axis around 1.0 so NAV=1.0 sits at the visual midpoint
_max_dev = max(abs(cum_returns.max() - 1.0), abs(cum_returns.min() - 1.0))
_pad = _max_dev * 0.15 if _max_dev > 0 else 0.01
_y_range = [1.0 - _max_dev - _pad, 1.0 + _max_dev + _pad]

fig_nav.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=11)),
    yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155",
               tickfont=dict(size=11), tickformat=".4f", range=_y_range),
    showlegend=False, hovermode="x unified"
)
st.plotly_chart(fig_nav, use_container_width=True)


# ── PERFORMANCE CHARTS ────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Performance Analysis</div>
    <div class="sec-sub">Daily P&L · Drawdown · Rolling Sharpe · Return Distribution</div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    colors = ["#34d399" if v > 0 else "#f87171" for v in net_pnl.values]
    fig_pnl = go.Figure(go.Bar(
        x=net_pnl.index, y=net_pnl.values * 100,
        marker_color=colors, opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra></extra>"
    ))
    fig_pnl.update_layout(
        title=dict(text="Daily P&L (%)", font=dict(color="#64748b", size=13)),
        height=250, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    fig_dist = go.Figure(go.Histogram(
        x=active.values * 100, nbinsx=40,
        marker_color="#38bdf8", opacity=0.7,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"
    ))
    fig_dist.add_vline(x=0, line_dash="dot", line_color="#334155", line_width=1)
    fig_dist.update_layout(
        title=dict(text="Distribution of Daily Returns", font=dict(color="#64748b", size=13)),
        height=250, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10), title="Daily Return (%)"),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_r:
    dd_series = (cum_returns / cum_returns.cummax() - 1) * 100
    fig_dd = go.Figure(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines", line=dict(color="#f87171", width=1.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.06)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>"
    ))
    fig_dd.add_hline(y=0, line_dash="dot", line_color="#1e2535", line_width=1)
    fig_dd.update_layout(
        title=dict(text="Drawdown from Peak (%)", font=dict(color="#64748b", size=13)),
        height=250, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    roll_sharpe = (
        net_pnl.rolling(63).mean() /
        (net_pnl.rolling(63).std() + 1e-8) * np.sqrt(252)
    )
    fig_rs = go.Figure(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        mode="lines", line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.04)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"
    ))
    fig_rs.add_hline(y=0, line_dash="dot", line_color="#1e2535", line_width=1)
    fig_rs.add_hline(y=1, line_dash="dot", line_color="rgba(52,211,153,0.25)", line_width=1)
    fig_rs.update_layout(
        title=dict(text="Rolling 63-Day Sharpe Ratio", font=dict(color="#64748b", size=13)),
        height=250, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_rs, use_container_width=True)

# Flow chart
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Pipeline Architecture</div>
    <div class="sec-sub">Transforming Raw Market Data to a Long-Short Portfolio</div>
</div>
<div class="flowchart">
    <div class="flow-box flow-box-active">
        <div class="flow-icon">📥</div>
        <div class="flow-label">Market Data</div>
        <div class="flow-sub">~100 S&P 500 stocks<br>daily prices 2021–2024</div>
    </div>
    <div class="flow-arrow">
        <div class="flow-arrow-inner">
            <div class="flow-arrow-line"></div>
            <div class="flow-arrow-head"></div>
        </div>
    </div>
    <div class="flow-box">
        <div class="flow-icon">🔬</div>
        <div class="flow-label">Factor Engineering</div>
        <div class="flow-sub">6 predictive signals<br>computed per stock per day</div>
    </div>
    <div class="flow-arrow">
        <div class="flow-arrow-inner">
            <div class="flow-arrow-line"></div>
            <div class="flow-arrow-head"></div>
        </div>
    </div>
    <div class="flow-box">
        <div class="flow-icon">🤖</div>
        <div class="flow-label">LightGBM Model</div>
        <div class="flow-sub">Learns which signals<br>matter most · Walk-forward</div>
    </div>
    <div class="flow-arrow">
        <div class="flow-arrow-inner">
            <div class="flow-arrow-line"></div>
            <div class="flow-arrow-head"></div>
        </div>
    </div>
    <div class="flow-box">
        <div class="flow-icon">📊</div>
        <div class="flow-label">Long-Short Portfolio</div>
        <div class="flow-sub">Long top 20%<br>Short bottom 20%</div>
    </div>
    <div class="flow-arrow">
        <div class="flow-arrow-inner">
            <div class="flow-arrow-line"></div>
            <div class="flow-arrow-head"></div>
        </div>
    </div>
    <div class="flow-box flow-box-active">
        <div class="flow-icon">✅</div>
        <div class="flow-label">Results</div>
        <div class="flow-sub">IC = 0.034 · ICIR = 0.83<br>Out-of-sample</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FACTOR ANALYSIS ───────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Alpha Factors — What the Model Learns From</div>
    <div class="sec-sub">Each factor is a measurable stock characteristic · IC measures predictive power</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<details class="info-details" style="margin-top:2px;">
    <summary>What is a factor?</summary>
    <div class="exp">
        A factor is a number computed for each stock on each day.
        For example, "Distance from 50-day Moving Average" tells us how far a stock's price has drifted 
        from its recent average. Stocks far below their average tend to bounce back — that's mean reversion.
        We measure each factor's predictive power with <strong>IC (Information Coefficient)</strong>: 
        the correlation between our rankings and actual future returns.
        IC > 0.02 is exploitable. We only feed the model factors with confirmed positive IC.
    </div>
</details>
""", unsafe_allow_html=True)

factors = [
    ("realized_vol", "Realized Volatility",       "High recent volatility predicts outperformance in our dataset",           0.033, True),
    ("ma50_dist",    "Distance from 50-day MA",   "Stocks below their 50-day average tend to mean-revert upward",            0.031, True),
    ("rsi",          "RSI Reversal",               "Oversold stocks (RSI < 50) tend to bounce — classic mean reversion",      0.013, True),
    ("ma20_dist",    "Distance from 20-day MA",   "Short-term mean reversion signal",                                        0.013, True),
    ("bollinger",    "Bollinger Band Z-Score",     "Price below the lower Bollinger Band is a statistically oversold signal", 0.006, True),
    ("rev_10d",      "10-Day Return Reversal",     "Stocks down over 10 consecutive days tend to recover",                    0.004, True),
    ("mom_1m",       "1-Month Momentum",           "Not predictive on S&P 500 large-caps at monthly horizon — excluded",      -0.017, False),
    ("breakout",     "52-Week High Breakout",      "Not a reliable signal in this dataset — excluded from model",             -0.030, False),
]

max_ic = max(abs(f[3]) for f in factors)

st.markdown('<div class="stattbl" style="padding:4px 0;">', unsafe_allow_html=True)
for code, name, desc, ic, used in factors:
    bar_w  = int(abs(ic) / max_ic * 100)
    bar_c  = "#34d399" if used else "#f87171"
    ic_c   = "#34d399" if used else "#f87171"
    ic_str = f"+{ic:.4f}" if ic > 0 else f"{ic:.4f}"
    badge  = f'<span class="ft-badge" style="background:rgba(52,211,153,0.08);color:#34d399;border:1px solid rgba(52,211,153,0.2);">✓ used</span>' if used else \
             f'<span class="ft-badge" style="background:rgba(248,113,113,0.08);color:#f87171;border:1px solid rgba(248,113,113,0.2);">✗ excl.</span>'
    st.markdown(f"""
    <div class="ftrow">
        <div class="ft-name">{code}</div>
        <div style="flex:1;">
            <div style="font-size:12px;color:#94a3b8;font-weight:500;">{name}</div>
            <div class="ft-desc">{desc}</div>
        </div>
        <div class="ft-bar-bg"><div class="ft-bar" style="width:{bar_w}%;background:{bar_c};"></div></div>
        <div class="ft-ic" style="color:{ic_c};">IC {ic_str}</div>
        {badge}
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── SHAP as Plotly chart ──────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head" style="margin-top:40px;">
    <div class="sec-title">SHAP Feature Importance</div>
    <div class="sec-sub">Which factors the model relied on most — averaged across all out-of-sample predictions</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<details class="info-details" style="margin-top:2px;">
    <summary>What is SHAP?</summary>
    <div class="exp">
        Instead of a black-box model, SHAP lets us see exactly 
        which factors drove each prediction. The chart below shows the average absolute contribution 
        of each factor across all out-of-sample predictions — the longer the bar, the more that 
        factor influenced the model's stock rankings.
    </div>
</details>
""", unsafe_allow_html=True)

shap_factors = ["realized_vol", "ma50_dist", "rsi", "ma20_dist", "bollinger", "rev_10d"]
shap_values  = [0.033, 0.028, 0.018, 0.015, 0.009, 0.006]

shap_df = pd.DataFrame({
    "factor": shap_factors,
    "importance": shap_values
}).sort_values("importance")

fig_shap = go.Figure(go.Bar(
    x=shap_df["importance"],
    y=shap_df["factor"],
    orientation="h",
    marker=dict(
        color=shap_df["importance"],
        colorscale=[[0, "#1e3a5f"], [0.5, "#38bdf8"], [1, "#7dd3fc"]],
        showscale=False
    ),
    hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
))
fig_shap.update_layout(
    height=260,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        showgrid=True, gridcolor="#0f1724",
        color="#334155", tickfont=dict(size=11),
        title=dict(text="Mean |SHAP value|", font=dict(color="#334155", size=11))
    ),
    yaxis=dict(
        showgrid=False, color="#7dd3fc",
        tickfont=dict(size=12, family="JetBrains Mono")
    )
)
st.plotly_chart(fig_shap, use_container_width=True)


# ── FULL STATS TABLE ──────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Full Statistics</div>
</div>
""", unsafe_allow_html=True)

daily_active = net_pnl[net_pnl != 0]
best_day  = daily_active.max() * 100
worst_day = daily_active.min() * 100
avg_win   = daily_active[daily_active > 0].mean() * 100
avg_loss  = daily_active[daily_active < 0].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(stat_table([
        ("Annualized Return",     f"{ann_ret:+.2f}%",   "#34d399" if ann_ret > 0 else "#f87171"),
        ("Annualized Volatility", f"{ann_vol:.2f}%",    "#94a3b8"),
        ("Sharpe Ratio",          f"{sharpe:.3f}",       "#34d399" if sharpe > 0 else "#f87171"),
        ("Calmar Ratio",          f"{calmar:.3f}",       "#94a3b8"),
        ("Max Drawdown",          f"{max_dd:.2f}%",      "#f87171"),
    ]), unsafe_allow_html=True)

with col2:
    st.markdown(stat_table([
        ("Win Rate",              f"{win_rate:.1f}%",   "#94a3b8"),
        ("Best Day",              f"+{best_day:.2f}%",  "#34d399"),
        ("Worst Day",             f"{worst_day:.2f}%",  "#f87171"),
        ("Avg Winning Day",       f"+{avg_win:.2f}%",   "#34d399"),
        ("Avg Losing Day",        f"{avg_loss:.2f}%",   "#f87171"),
    ]), unsafe_allow_html=True)

with col3:
    st.markdown(stat_table([
        ("Model Information Coefficient",              "0.0343",              "#34d399"),
        ("ICIR",                  "0.83",                "#34d399"),
        ("Universe",              "~100 S&P 500",        "#94a3b8"),
        ("Backtest Period",       "2022–2024",           "#94a3b8"),
        ("Active Trading Days",   f"{len(daily_active)}", "#94a3b8"),
    ]), unsafe_allow_html=True)


# ── TECHNICAL SPEC ────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Technical Specification</div>
    <div class="sec-sub">Model Hyperparameters and Backtest Parameters</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="codebox">
        <div style="color:#475569;font-size:10px;margin-bottom:12px;letter-spacing:0.1em;">LIGHTGBM MODEL</div>
        <span class="key">model</span>             LGBMRegressor<br>
        <span class="key">n_estimators</span>      200<br>
        <span class="key">learning_rate</span>     0.02<br>
        <span class="key">num_leaves</span>        31<br>
        <span class="key">min_child_samples</span> 50<br>
        <span class="key">reg_lambda</span>        1.0<br>
        <span class="key">reg_alpha</span>         0.1<br>
        <span class="key">subsample</span>         0.8<br>
        <span class="key">colsample_bytree</span>  0.8<br>
        <span class="key">objective</span>         regression (IC-ranked)<br>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="codebox">
        <div style="color:#475569;font-size:10px;margin-bottom:12px;letter-spacing:0.1em;">BACKTEST PARAMETERS</div>
        <span class="val">data</span>              S&P 500 large-cap equities<br>
        <span class="val">train_window</span>      252 trading days (1 year)<br>
        <span class="val">test_window</span>       63 trading days (3 months)<br>
        <span class="val">step_size</span>         21 trading days (1 month)<br>
        <span class="val">forward_days</span>      21 (predict 1-month returns)<br>
        <span class="val">long_pct</span>          top 20% by model score<br>
        <span class="val">short_pct</span>         bottom 20% by model score<br>
        <span class="val">rebal_freq</span>        monthly<br>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Alpha Factor Discovery &nbsp;·&nbsp;
    Copyright 2026 Prayag Gaonkar. All rights reserved.
</div>
""", unsafe_allow_html=True)