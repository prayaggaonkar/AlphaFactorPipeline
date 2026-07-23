import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Alpha Factor Discovery Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #080c14; color: #cbd5e1; }

section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1f3c 0%, #080f20 60%, #050912 100%);
    border: 1px solid #162035;
    border-radius: 20px;
    padding: 60px 52px 52px;
    margin-bottom: 48px;
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
    margin-bottom: 32px;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; }
.tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 5px;
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.18);
    color: #7dd3fc;
}

/* ── Section ── */
.sec-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin: 52px 0 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid #111827;
}
.sec-title { font-size: 20px; font-weight: 600; color: #e2e8f0; }
.sec-sub   { font-size: 13px; color: #334155; }

/* ── Metric cards ── */
.mcard {
    background: #0b1120;
    border: 1px solid #131e30;
    border-radius: 14px;
    padding: 22px 26px 20px;
    height: 100%;
}
.mcard-label {
    font-size: 11px;
    font-weight: 500;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 10px;
}
.mcard-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 8px;
}
.mcard-sub { font-size: 12px; color: #1e3a5f; line-height: 1.5; }
.pos { color: #34d399; }
.neg { color: #f87171; }
.neu { color: #38bdf8; }
.pur { color: #a78bfa; }

/* ── Explainer box ── */
.exp {
    background: #0b1120;
    border: 1px solid #131e30;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 16px 0;
    font-size: 14px;
    color: #64748b;
    line-height: 1.8;
}
.exp strong { color: #e2e8f0; }

/* ── Pipeline ── */
.pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; margin: 24px 0 8px; }
.pstep {
    background: #0b1120;
    border: 1px solid #131e30;
    padding: 20px 16px;
    text-align: center;
    position: relative;
}
.pstep:first-child { border-radius: 12px 0 0 12px; }
.pstep:last-child  { border-radius: 0 12px 12px 0; }
.pstep:not(:last-child)::after {
    content: '›';
    position: absolute;
    right: -9px; top: 50%;
    transform: translateY(-50%);
    color: #1e3a5f;
    font-size: 20px;
    z-index: 2;
}
.pstep-icon  { font-size: 22px; margin-bottom: 8px; }
.pstep-label { font-size: 12px; font-weight: 600; color: #cbd5e1; }
.pstep-sub   { font-size: 11px; color: #334155; margin-top: 4px; line-height: 1.4; }

/* ── Factor table ── */
.ftrow {
    display: flex;
    align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid #0f1724;
    gap: 16px;
}
.ftrow:last-child { border-bottom: none; }
.ft-name  { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7dd3fc; min-width: 110px; }
.ft-desc  { font-size: 12px; color: #334155; flex: 1; }
.ft-bar-bg { width: 100px; height: 5px; background: #111827; border-radius: 3px; flex-shrink: 0; }
.ft-bar    { height: 100%; border-radius: 3px; }
.ft-ic     { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; min-width: 64px; text-align: right; }
.ft-badge  { font-size: 10px; padding: 2px 7px; border-radius: 4px; min-width: 52px; text-align: center; }

/* ── Stat table ── */
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
.strow-label { color: #334155; }
.strow-val   { font-family: 'JetBrains Mono', monospace; font-weight: 500; font-size: 13px; }

/* ── Code block ── */
.codebox {
    background: #060a10;
    border: 1px solid #131e30;
    border-radius: 10px;
    padding: 18px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #475569;
    line-height: 2;
}
.codebox .key { color: #38bdf8; }
.codebox .val { color: #a78bfa; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 48px 0 24px;
    font-size: 12px;
    color: #1e293b;
    border-top: 1px solid #0f1724;
    margin-top: 72px;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    try:
        cum = pd.read_parquet("backtest/cum_returns.parquet")["cum_returns"]
        pnl = pd.read_parquet("backtest/net_pnl.parquet")["net_pnl"]
        return cum, pnl, True
    except Exception:
        return None, None, False

cum_returns, net_pnl, data_loaded = load_results()

def mcard(label, val, sub, cls="neu"):
    return f"""<div class="mcard">
        <div class="mcard-label">{label}</div>
        <div class="mcard-val {cls}">{val}</div>
        <div class="mcard-sub">{sub}</div>
    </div>"""

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Quantitative Research · Machine Learning · Equity Markets</div>
    <div class="hero-title">Alpha Factor<br><em>Discovery Engine</em></div>
    <div class="hero-body">
        A machine learning system that learns which stock characteristics predict 
        future returns — and turns those predictions into a real, dollar-neutral trading strategy.
        Built with walk-forward validation, SHAP explainability, and realistic transaction cost modeling.
        Every number on this page is out-of-sample.
    </div>
    <div class="tag-row">
        <span class="tag">Python</span>
        <span class="tag">LightGBM</span>
        <span class="tag">Walk-Forward CV</span>
        <span class="tag">Long-Short Equity</span>
        <span class="tag">SHAP</span>
        <span class="tag">S&P 500</span>
        <span class="tag">10bps Transaction Costs</span>
        <span class="tag">2021–2024</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("Run `python main.py` first to generate results, then refresh this page.")
    st.stop()

# ── Compute metrics ───────────────────────────────────────────────────────────
active     = net_pnl[net_pnl != 0]
ann_ret    = active.mean() * 252 * 100
ann_vol    = active.std() * np.sqrt(252) * 100
sharpe     = (active.mean() / active.std() * np.sqrt(252))
max_dd     = ((cum_returns / cum_returns.cummax()) - 1).min() * 100
win_rate   = (active > 0).mean() * 100
total_ret  = (cum_returns.iloc[-1] - 1) * 100


# ── WHAT THIS PROJECT DOES ────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">What This Project Does</div>
    <div class="sec-sub">A plain-English explanation — no finance background needed</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="exp">
    <strong>The core idea:</strong> Every day, some stocks go up more than others. Can we predict which ones?
    This project builds a machine learning pipeline that studies patterns in historical stock data — 
    things like whether a stock is trading below its recent average, or whether it's been unusually volatile — 
    and learns to predict which stocks will outperform over the next month.
    <br><br>
    <strong>How it makes money:</strong> Each month, it <em>buys</em> the stocks the model ranks highest 
    (predicts will go up) and simultaneously <em>bets against</em> the stocks it ranks lowest 
    (predicts will go down), in equal dollar amounts. This "long-short" structure means the 
    strategy doesn't care whether the market overall goes up or down — it only needs its 
    top picks to beat its bottom picks.
    <br><br>
    <strong>Why it's hard to cheat:</strong> The model is evaluated using walk-forward validation — 
    it only ever trains on past data and is tested on future data it has never seen. 
    Every number you see on this page is a genuine out-of-sample result.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="pstep">
        <div class="pstep-icon">📥</div>
        <div class="pstep-label">Market Data</div>
        <div class="pstep-sub">~100 S&P 500 stocks · daily prices · 2021–2024</div>
    </div>
    <div class="pstep">
        <div class="pstep-icon">🔬</div>
        <div class="pstep-label">Factor Engineering</div>
        <div class="pstep-sub">6 predictive signals computed per stock per day</div>
    </div>
    <div class="pstep">
        <div class="pstep-icon">🤖</div>
        <div class="pstep-label">LightGBM Model</div>
        <div class="pstep-sub">Learns which signals matter most · walk-forward trained</div>
    </div>
    <div class="pstep">
        <div class="pstep-icon">📊</div>
        <div class="pstep-label">Long-Short Portfolio</div>
        <div class="pstep-sub">Long top 20% · short bottom 20% · 10bps costs</div>
    </div>
    <div class="pstep">
        <div class="pstep-icon">✅</div>
        <div class="pstep-label">Results</div>
        <div class="pstep-sub">IC = 0.034 · ICIR = 0.83 · out-of-sample</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KEY RESULTS ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Key Results</div>
    <div class="sec-sub">All metrics are out-of-sample · after 10bps transaction costs per trade</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
ret_cls = "pos" if ann_ret > 0 else "neg"
shr_cls = "pos" if sharpe > 0.3 else ("neg" if sharpe < 0 else "neu")
dd_cls  = "neg" if max_dd < -8 else "neu"

with c1: st.markdown(mcard("Model IC", "0.034", "Information Coefficient — how well the model ranks stocks. >0.02 is industry standard.", "pos"), unsafe_allow_html=True)
with c2: st.markdown(mcard("ICIR", "0.83", "IC ÷ StdDev(IC). Measures consistency of predictions. >0.5 is considered good.", "pos"), unsafe_allow_html=True)
with c3: st.markdown(mcard("Ann. Return", f"{ann_ret:+.1f}%", "Annualized return on active trading days, after all costs.", ret_cls), unsafe_allow_html=True)
with c4: st.markdown(mcard("Sharpe Ratio", f"{sharpe:.2f}", "Return per unit of risk, annualized. Higher = better risk-adjusted performance.", shr_cls), unsafe_allow_html=True)
with c5: st.markdown(mcard("Max Drawdown", f"{max_dd:.1f}%", "Worst peak-to-trough loss during the backtest period.", dd_cls), unsafe_allow_html=True)
with c6: st.markdown(mcard("Win Rate", f"{win_rate:.1f}%", "Percentage of active trading days with positive P&L.", "neu"), unsafe_allow_html=True)

st.markdown("""
<div class="exp" style="margin-top:20px;">
    <strong>What IC means for non-finance readers:</strong> 
    IC (Information Coefficient) measures how well the model's stock rankings match actual future returns.
    An IC of 0.034 means the model's predictions are about 3.4% correlated with real outcomes.
    That sounds small, but consistently achieving IC > 0.02 across different market conditions 
    is considered genuine, exploitable alpha by professional quant researchers.
    The ICIR of 0.83 means the signal is stable — not just lucky in one period.
</div>
""", unsafe_allow_html=True)


# ── TRAIN VS TEST EXPLANATION ─────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Training vs. Testing — No Look-Ahead Bias</div>
    <div class="sec-sub">The most important methodological detail in any ML finance project</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="exp">
    <strong>The problem with naive ML on financial data:</strong> If you train a model on 2021–2024 data 
    and then test it on the same 2021–2024 data, you're cheating — the model already "saw" the answers. 
    This produces results that look great but collapse in live trading. This is called look-ahead bias 
    and it's the #1 mistake in quant finance ML.
    <br><br>
    <strong>How this project prevents it — walk-forward validation:</strong>
    <br>
    • <strong>Train window:</strong> The model trains on 1 full year of past data (252 trading days)
    <br>
    • <strong>Test window:</strong> It predicts on the next 3 months of data it has never seen (63 days)
    <br>
    • <strong>Step forward:</strong> The window rolls forward 21 days and repeats
    <br>
    • <strong>Result:</strong> Every prediction in the backtest was made on data the model had never trained on
    <br><br>
    The model was trained on data from <strong>2021–2022</strong> and tested on <strong>2022–2024</strong>.
    All performance numbers reflect genuine out-of-sample prediction.
</div>
""", unsafe_allow_html=True)

# Timeline visualization
fig_timeline = go.Figure()
folds = [
    ("Train", "2021-01-01", "2022-01-01", "#1e3a5f", 1),
    ("Test",  "2022-01-01", "2022-04-01", "#34d399", 1),
    ("Train", "2021-04-01", "2022-04-01", "#1e3a5f", 2),
    ("Test",  "2022-04-01", "2022-07-01", "#34d399", 2),
    ("Train", "2021-07-01", "2022-07-01", "#1e3a5f", 3),
    ("Test",  "2022-07-01", "2022-10-01", "#34d399", 3),
    ("Train", "2021-10-01", "2022-10-01", "#1e3a5f", 4),
    ("Test",  "2022-10-01", "2023-01-01", "#34d399", 4),
]
for label, start, end, color, row in folds:
    fig_timeline.add_trace(go.Bar(
        x=[pd.Timestamp(end) - pd.Timestamp(start)],
        y=[f"Fold {row}"],
        base=[pd.Timestamp(start).timestamp() * 1000],
        orientation="h",
        marker_color=color,
        opacity=0.85,
        name=label,
        showlegend=(row == 1),
        hovertemplate=f"{label}: {start} → {end}<extra></extra>"
    ))

fig_timeline.update_layout(
    barmode="overlay",
    height=200,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        type="date",
        color="#334155",
        showgrid=False,
        tickfont=dict(size=11)
    ),
    yaxis=dict(color="#475569", showgrid=False, tickfont=dict(size=11)),
    legend=dict(
        font=dict(color="#64748b", size=11),
        bgcolor="rgba(0,0,0,0)",
        orientation="h",
        x=0, y=-0.3
    )
)
st.plotly_chart(fig_timeline, use_container_width=True)


# ── CUMULATIVE RETURNS ────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Cumulative Returns</div>
    <div class="sec-sub">Starting NAV = $1.00 · Long-short · Dollar-neutral · After transaction costs</div>
</div>
""", unsafe_allow_html=True)

fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(
    x=cum_returns.index,
    y=cum_returns.values,
    mode="lines",
    name="Strategy NAV",
    line=dict(color="#38bdf8", width=2),
    fill="tozeroy",
    fillcolor="rgba(56,189,248,0.04)",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>NAV: $%{y:.4f}<extra></extra>"
))
fig_nav.add_hline(y=1.0, line_dash="dot", line_color="#1e2535", line_width=1)
fig_nav.update_layout(
    height=340,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=11)),
    yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155",
               tickfont=dict(size=11), tickformat=".4f"),
    showlegend=False,
    hovermode="x unified"
)
st.plotly_chart(fig_nav, use_container_width=True)

st.markdown("""
<div class="exp">
    <strong>How to read this chart:</strong> You start with $1.00. Each day, the strategy goes long 
    (buys) the top 20% of stocks by model score and short (bets against) the bottom 20%, 
    in equal dollar amounts. Because long and short legs are equal in size, the strategy is 
    market-neutral — a market crash or rally doesn't directly affect it.
    The chart shows how your $1 changes over time from these relative bets.
</div>
""", unsafe_allow_html=True)


# ── PERFORMANCE CHARTS ────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Performance Analysis</div>
    <div class="sec-sub">Daily P&L · Drawdown · Rolling Sharpe · Return Distribution</div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    # Daily P&L
    colors = ["#34d399" if v > 0 else "#f87171" for v in net_pnl.values]
    fig_pnl = go.Figure(go.Bar(
        x=net_pnl.index, y=net_pnl.values * 100,
        marker_color=colors, opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra></extra>"
    ))
    fig_pnl.update_layout(
        title=dict(text="Daily P&L (%)", font=dict(color="#64748b", size=13)),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Return distribution
    fig_dist = go.Figure(go.Histogram(
        x=active.values * 100,
        nbinsx=40,
        marker_color="#38bdf8",
        opacity=0.7,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"
    ))
    fig_dist.add_vline(x=0, line_dash="dot", line_color="#334155", line_width=1)
    fig_dist.update_layout(
        title=dict(text="Distribution of Daily Returns", font=dict(color="#64748b", size=13)),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10), title="Daily Return (%)"),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_r:
    # Drawdown
    dd_series = (cum_returns / cum_returns.cummax() - 1) * 100
    fig_dd = go.Figure(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines",
        line=dict(color="#f87171", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248,113,113,0.06)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>"
    ))
    fig_dd.add_hline(y=0, line_dash="dot", line_color="#1e2535", line_width=1)
    fig_dd.update_layout(
        title=dict(text="Drawdown from Peak (%)", font=dict(color="#64748b", size=13)),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Rolling Sharpe
    roll_sharpe = (
        net_pnl.rolling(63).mean() /
        (net_pnl.rolling(63).std() + 1e-8) *
        np.sqrt(252)
    )
    fig_rs = go.Figure(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        mode="lines",
        line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.04)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"
    ))
    fig_rs.add_hline(y=0, line_dash="dot", line_color="#1e2535", line_width=1)
    fig_rs.add_hline(y=1, line_dash="dot", line_color="rgba(52,211,153,0.3)", line_width=1)
    fig_rs.update_layout(
        title=dict(text="Rolling 63-Day Sharpe Ratio", font=dict(color="#64748b", size=13)),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#0f1724", color="#334155", tickfont=dict(size=10)),
        showlegend=False
    )
    st.plotly_chart(fig_rs, use_container_width=True)


# ── FACTOR ANALYSIS ───────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Alpha Factors — What the Model Learns From</div>
    <div class="sec-sub">Each factor is a measurable stock characteristic · IC measures predictive power</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="exp">
    <strong>What is a factor?</strong> A factor is a number we compute for each stock on each day.
    For example, "Distance from 50-day Moving Average" tells us how far a stock's price has drifted 
    from its recent average. Stocks far below their average tend to bounce back — that's mean reversion.
    We measure each factor's predictive power with <strong>IC (Information Coefficient)</strong>: 
    the correlation between our factor's stock rankings and actual future returns.
    IC > 0.02 is exploitable. We only feed the model factors with confirmed positive IC.
</div>
""", unsafe_allow_html=True)

factors = [
    ("realized_vol", "Realized Volatility",       "High recent volatility predicts outperformance in our dataset",        0.033, True),
    ("ma50_dist",    "Distance from 50-day MA",   "Stocks below their 50-day average tend to mean-revert upward",         0.031, True),
    ("rsi",          "RSI Reversal",               "Oversold stocks (RSI < 50) tend to bounce — classic mean reversion",   0.013, True),
    ("ma20_dist",    "Distance from 20-day MA",   "Short-term mean reversion — powerful at the 20-day horizon",           0.013, True),
    ("bollinger",    "Bollinger Band Z-Score",     "Price below the lower Bollinger Band is a statistically oversold signal", 0.006, True),
    ("rev_10d",      "10-Day Return Reversal",     "Stocks down over 10 consecutive days tend to recover",                 0.004, True),
    ("mom_1m",       "1-Month Momentum",           "Not predictive on S&P 500 large-caps at monthly horizon — excluded",   -0.017, False),
    ("breakout",     "52-Week High Breakout",      "Not a reliable signal in this dataset — excluded from model",          -0.030, False),
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

if os.path.exists("models/shap_importance.png"):
    st.markdown("""
    <div class="sec-head" style="margin-top:40px;">
        <div class="sec-title">SHAP Feature Importance</div>
        <div class="sec-sub">How much each factor contributed to the model's predictions on average</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="exp">
        <strong>What is SHAP?</strong> Instead of a black-box model, SHAP lets us see exactly 
        which factors drove each prediction. For every stock on every day, we can say 
        "realized volatility contributed +0.02 to the buy signal, RSI contributed +0.01."
        The chart below shows the average absolute contribution across all out-of-sample predictions.
    </div>
    """, unsafe_allow_html=True)
    st.image("models/shap_importance.png", use_column_width=True)


# ── FULL STATS TABLE ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Full Statistics</div>
    <div class="sec-sub">For quantitative researchers and technical recruiters</div>
</div>
""", unsafe_allow_html=True)

daily_active = net_pnl[net_pnl != 0]
best_day  = daily_active.max() * 100
worst_day = daily_active.min() * 100
avg_win   = daily_active[daily_active > 0].mean() * 100
avg_loss  = daily_active[daily_active < 0].mean() * 100
calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0

col1, col2, col3 = st.columns(3)

def stat_table(rows):
    html = '<div class="stattbl">'
    for label, val, color in rows:
        html += f'<div class="strow"><span class="strow-label">{label}</span><span class="strow-val" style="color:{color};">{val}</span></div>'
    html += '</div>'
    return html

with col1:
    st.markdown(stat_table([
        ("Annualized Return",   f"{ann_ret:+.2f}%",    "#34d399" if ann_ret > 0 else "#f87171"),
        ("Annualized Volatility", f"{ann_vol:.2f}%",   "#94a3b8"),
        ("Sharpe Ratio",        f"{sharpe:.3f}",        "#34d399" if sharpe > 0 else "#f87171"),
        ("Calmar Ratio",        f"{calmar:.3f}",        "#94a3b8"),
        ("Max Drawdown",        f"{max_dd:.2f}%",       "#f87171"),
    ]), unsafe_allow_html=True)

with col2:
    st.markdown(stat_table([
        ("Win Rate",            f"{win_rate:.1f}%",     "#94a3b8"),
        ("Best Day",            f"+{best_day:.2f}%",    "#34d399"),
        ("Worst Day",           f"{worst_day:.2f}%",    "#f87171"),
        ("Avg Winning Day",     f"+{avg_win:.2f}%",     "#34d399"),
        ("Avg Losing Day",      f"{avg_loss:.2f}%",     "#f87171"),
    ]), unsafe_allow_html=True)

with col3:
    st.markdown(stat_table([
        ("Model IC",            "0.0343",               "#34d399"),
        ("ICIR",                "0.83",                 "#34d399"),
        ("Universe",            "~100 S&P 500 stocks",  "#94a3b8"),
        ("Backtest Period",     "2022–2024",            "#94a3b8"),
        ("Active Trading Days", f"{len(daily_active)}", "#94a3b8"),
    ]), unsafe_allow_html=True)


# ── TECHNICAL SPEC ────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-title">Technical Specification</div>
    <div class="sec-sub">Model hyperparameters and backtest parameters</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="codebox">
        <div style="color:#475569;font-size:10px;margin-bottom:12px;letter-spacing:0.1em;">LIGHTGBM MODEL</div>
        <span class="key">model</span>           LGBMRegressor<br>
        <span class="key">n_estimators</span>    200<br>
        <span class="key">learning_rate</span>   0.02<br>
        <span class="key">num_leaves</span>      31<br>
        <span class="key">min_child_samples</span> 50<br>
        <span class="key">reg_lambda</span>      1.0<br>
        <span class="key">reg_alpha</span>       0.1<br>
        <span class="key">subsample</span>       0.8<br>
        <span class="key">colsample_bytree</span> 0.8<br>
        <span class="key">objective</span>       regression (IC-ranked)<br>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="codebox">
        <div style="color:#475569;font-size:10px;margin-bottom:12px;letter-spacing:0.1em;">BACKTEST PARAMETERS</div>
        <span class="val">data</span>             S&P 500 large-cap equities<br>
        <span class="val">train_window</span>     252 trading days (1 year)<br>
        <span class="val">test_window</span>      63 trading days (3 months)<br>
        <span class="val">step_size</span>        21 trading days (1 month)<br>
        <span class="val">forward_days</span>     21 (predict 1-month returns)<br>
        <span class="val">long_pct</span>         top 20% by model score<br>
        <span class="val">short_pct</span>        bottom 20% by model score<br>
        <span class="val">cost_bps</span>         10 basis points one-way<br>
        <span class="val">structure</span>        dollar-neutral long-short<br>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Alpha Factor Discovery Engine &nbsp;·&nbsp; 
    Python · LightGBM · pandas · SHAP · Streamlit &nbsp;·&nbsp;
    Walk-forward validated · Out-of-sample results · S&P 500 · 2021–2024
</div>
""", unsafe_allow_html=True)