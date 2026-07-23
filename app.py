import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Alpha Factor Discovery Pipeline",
    page_icon="📈",
    layout="wide"
)


# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #666;
        margin-bottom: 30px;
    }

    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
    }

    .metric-label {
        font-size: 14px;
        color: gray;
    }

    .metric-value {
        font-size: 30px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
    📈 Alpha Factor Discovery Pipeline
    </div>

    <div class="subtitle">
    A machine learning framework for discovering stock market signals 
    and evaluating their ability to generate excess returns.
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------
# PROJECT DESCRIPTION
# -------------------------------------------------------

with st.container():

    st.markdown("## About This Project")

    st.write(
        """
        This project builds an end-to-end quantitative research pipeline 
        to discover and evaluate stock market factors.

        Traditional investors search for **alpha** — signals that can predict 
        future stock performance beyond what the overall market provides.

        This system:

        - Collects historical market data
        - Generates quantitative factors
        - Normalizes and processes signals
        - Trains machine learning models
        - Performs walk-forward backtesting
        - Evaluates portfolio performance

        The goal is not to predict exact stock prices, but to rank stocks 
        based on their probability of outperforming the market.
        """
    )



# -------------------------------------------------------
# PIPELINE EXPLANATION
# -------------------------------------------------------

st.markdown("---")
st.markdown("## Research Pipeline")


pipeline_cols = st.columns(5)

steps = [
    ("1", "Market Data", "Historical prices & fundamentals"),
    ("2", "Factor Engine", "Momentum, volatility, technical signals"),
    ("3", "ML Model", "Learns relationships between factors and returns"),
    ("4", "Portfolio", "Long strongest signals / short weakest signals"),
    ("5", "Backtest", "Evaluate historical performance")
]


for col, step in zip(pipeline_cols, steps):

    col.markdown(
        f"""
        <div class="card">

        <h3>{step[0]}</h3>

        <b>{step[1]}</b>

        <br><br>

        {step[2]}

        </div>
        """,
        unsafe_allow_html=True
    )



# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("Dashboard Controls")

st.sidebar.info(
    """
    Adjust settings and explore model performance.

    Results are generated after running:
    
    `python main.py`
    """
)


top_n = st.sidebar.slider(
    "Long/Short Portfolio Size",
    5,
    30,
    10
)



# -------------------------------------------------------
# LOAD RESULTS
# -------------------------------------------------------

try:

    cum_ret = pd.read_parquet(
        "backtest/cum_returns.parquet"
    )

    net_pnl = pd.read_parquet(
        "backtest/net_pnl.parquet"
    )


except FileNotFoundError:

    st.error(
        """
        Backtest results not found.

        Run:

        ```
        python main.py
        ```

        before launching Streamlit.
        """
    )

    st.stop()



# -------------------------------------------------------
# CALCULATE METRICS
# -------------------------------------------------------

returns = net_pnl.squeeze()

equity = cum_ret.squeeze()


annual_return = (
    returns.mean()
    *252
)

volatility = (
    returns.std()
    *np.sqrt(252)
)


sharpe = (
    annual_return /
    volatility
)


max_drawdown = (
    equity /
    equity.cummax()
    -1
).min()


win_rate = (
    returns > 0
).mean()



# -------------------------------------------------------
# PERFORMANCE SUMMARY
# -------------------------------------------------------

st.markdown("---")
st.markdown("## Performance Summary")


m1,m2,m3,m4 = st.columns(4)


m1.metric(
    "Annualized Return",
    f"{annual_return*100:.2f}%"
)

m2.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}"
)


m3.metric(
    "Maximum Drawdown",
    f"{max_drawdown*100:.2f}%"
)


m4.metric(
    "Winning Days",
    f"{win_rate*100:.1f}%"
)



# -------------------------------------------------------
# EQUITY CURVE
# -------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    ## Portfolio Growth

    This chart shows how a hypothetical $1 investment 
    would have grown over the backtest period.
    """
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=equity.index,
        y=equity,
        mode="lines",
        name="Strategy"
    )
)


fig.update_layout(
    height=450,
    template="plotly_white",
    yaxis_title="Portfolio Value",
    xaxis_title="Date"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# -------------------------------------------------------
# DRAWDOWN
# -------------------------------------------------------

st.markdown(
    """
    ## Risk Analysis

    Drawdown measures the decline from the portfolio's 
    previous peak. Smaller drawdowns indicate better risk control.
    """
)


dd = equity/equity.cummax()-1


fig_dd = px.area(
    dd,
    title="Portfolio Drawdown"
)


fig_dd.update_layout(
    template="plotly_white",
    yaxis_title="Drawdown"
)


st.plotly_chart(
    fig_dd,
    use_container_width=True
)



# -------------------------------------------------------
# ROLLING SHARPE
# -------------------------------------------------------

st.markdown(
    """
    ## Strategy Consistency

    Rolling Sharpe shows whether the strategy maintains 
    risk-adjusted performance across different market periods.
    """
)


rolling_sharpe = (
    returns
    .rolling(63)
    .mean()
    /
    returns
    .rolling(63)
    .std()
    *
    np.sqrt(252)
)


fig_rs = px.line(
    rolling_sharpe,
    title="63-Day Rolling Sharpe Ratio"
)


fig_rs.add_hline(
    y=0,
    line_dash="dash"
)


fig_rs.update_layout(
    template="plotly_white"
)


st.plotly_chart(
    fig_rs,
    use_container_width=True
)



# -------------------------------------------------------
# RETURNS DISTRIBUTION
# -------------------------------------------------------

st.markdown(
    """
    ## Return Distribution

    Shows the frequency of daily portfolio returns.
    """
)


fig_hist = px.histogram(
    returns,
    nbins=50,
    title="Daily Returns"
)


fig_hist.update_layout(
    template="plotly_white"
)


st.plotly_chart(
    fig_hist,
    use_container_width=True
)




# -------------------------------------------------------
# TECHNICAL DETAILS
# -------------------------------------------------------

st.markdown("---")

with st.expander(
    "🔬 Technical Details (For Recruiters / Quant Researchers)"
):

    st.markdown(
        """
        ### Model Architecture

        **Data Processing**
        - Historical equity price data
        - Feature engineering
        - Factor normalization
        - Outlier handling


        ### Portfolio Construction

        Stocks are ranked based on predicted future performance.

        Portfolio methodology:

        - Long: highest ranked stocks
        - Short: lowest ranked stocks
        - Market neutral exposure


        ### Backtesting Methodology

        Walk-forward validation:

        - Train period: historical window
        - Prediction period: future returns
        - Rolling evaluation prevents look-ahead bias


        ### Evaluation Metrics

        - Annualized Return
        - Sharpe Ratio
        - Maximum Drawdown
        - Win Rate
        - Rolling Risk Adjusted Performance

        """
    )



st.success(
    "Alpha Factor Pipeline successfully loaded."
)