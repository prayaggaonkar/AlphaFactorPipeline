## main.py  — run this file to execute the whole project
from data_loader import load_prices, compute_returns
from factor_lib  import build_factor_matrix
from model       import train_and_evaluate
from backtest    import build_portfolio
import yfinance as yf
from config import UNIVERSE_SIZE, START_DATE, END_DATE

