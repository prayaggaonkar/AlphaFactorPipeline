## main.py  — run this file to execute the whole project
from data_loader import load_prices, compute_returns
from factor_lib  import build_factor_matrix
from model       import train_and_evaluate
from backtest    import build_portfolio
import yfinance as yf
from config import UNIVERSE_SIZE, START_DATE, END_DATE

print("=== Step 1: Load data ===")
prices = load_prices()      # run data_loader.py first to download
volume = yf.download(
    list(prices.columns), start=START_DATE, end=END_DATE,
    auto_adjust=True, progress=False
)["Volume"].reindex(columns=prices.columns)