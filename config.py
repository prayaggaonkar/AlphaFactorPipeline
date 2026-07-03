## config.py
## Change these values to experiment with different setups

UNIVERSE_SIZE = 50        # number of stocks (start small, expand later)
START_DATE    = "2024-01-01"
END_DATE      = "2025-01-01"
FORWARD_DAYS  = 5         # predict 5-day future returns
REBAL_FREQ    = 5         # rebalance every 5 trading days
TRAIN_WINDOW  = 252       # 1 year of training data
TEST_WINDOW   = 63        # test on 3 months at a time
COST_BPS      = 10        # transaction cost in basis points (0.10%)
DATA_DIR      = "data/"
FACTOR_DIR    = "factors/"