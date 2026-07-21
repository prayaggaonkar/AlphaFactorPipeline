## config.py
## Change these values to experiment with different setups

UNIVERSE_SIZE = 150
START_DATE    = "2018-01-01"
END_DATE      = "2024-01-01"
FORWARD_DAYS  = 21        # 1 month
REBAL_FREQ    = 21        # rebalance monthly to match prediction horizon
TRAIN_WINDOW  = 504       # 2 years of training data
TEST_WINDOW   = 63
COST_BPS      = 10
DATA_DIR      = "data/"
FACTOR_DIR    = "factors/"