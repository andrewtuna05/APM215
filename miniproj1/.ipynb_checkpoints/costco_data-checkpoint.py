import numpy as numpy
import pandas as pd
import yfinance as yf
import datetime


# Seasonal Year 1
start_date_1 = datetime.datetime(2024, 9, 1)
end_date_1 = datetime.datetime(2024, 10, 31)

# Seasonal Year 2
start_date_2 = datetime.datetime(2025, 9, 1)
end_date_2 = datetime.datetime(2025, 10, 31)

data = yf.download("COST")

