# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats

def diagnose_data(df):
    """
    Diagnose data to detect any potential errors.
    """
    print("\nThe total numbers of lines:")
    print(len(df))

    print("\nData types:")
    print(df.dtypes)

    print("\nNull value summary:")
    print(df.isnull().sum())

    print("\nHead of the dataframe:")
    print(df.head())

def add_technical_indicators(df):
  """
  Add technical indicators to the dataframe.
  """
  df = df.copy()
  
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0) # handle multiple columns
    
  df.columns = [str(col).strip() for col in df.columns]
  
  if isinstance(df.index, pd.DatetimeIndex):
    df['Date'] = df.index
  df = df.reset_index(drop=True)
  
  if 'Ticker' not in df.columns:
        df['Ticker'] = 'SINGLE'
  
  # check required columns
  required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
      raise KeyError(f"Missing required columns: {missing_columns}") 
    
  try:
    df = df.sort_values(['Ticker', 'Date'])
    
    # this function is generated with the help of ChatGPT
    def calculate_indicators(group):
      """
      Add technical indicators
      """
      group = group.copy()
      group['RSI'] = ta.rsi(group['Close'], length=14)
      macd = ta.macd(group['Close'], fast=12, slow=26, signal=9)
      if macd is not None:
          group['MACD'] = macd['MACD_12_26_9']
          group['MACD_signal'] = macd['MACDs_12_26_9']
      
      bollinger = ta.bbands(group['Close'], length=20)
      if bollinger is not None:
          group['BB_upper'] = bollinger['BBU_20_2.0']
          group['BB_lower'] = bollinger['BBL_20_2.0']
      
      group['MA_20'] = ta.sma(group['Close'], length=20)
      group['SMA_50'] = ta.sma(group['Close'], length=50)
      
      return group
        
    # apply calculations to each group
    df = df.groupby('Ticker', group_keys=False).apply(calculate_indicators)
    return df
      
  except Exception as e:
    print(f"Error calculating technical indicators: {str(e)}")
    return None

def compute_market_trend(df):
  """
  Calculate the overall market trend by averaging tech indicators across all tickers.
  """
  # handle NaN values
  if df is None or df.empty:
    return None
  df = df.ffill().bfill() 
  
  mk_trend = df.groupby('Date').agg({
        'Close': ['mean', 'std', 'min', 'max'],
        'Volume': ['mean', 'sum'],
        'RSI': 'mean',
        'MACD': 'mean',
        'MACD_signal': 'mean',
        'BB_upper': 'mean',
        'BB_lower': 'mean',
        'MA_20': 'mean',
        'SMA_50': 'mean'
    }).reset_index()

  # name columns
  mk_trend.columns = ['Date',
                        'Avg_Close_Price', 'Close_Price_Volatility', 
                        'Min_Close_Price', 'Max_Close_Price',
                        'Avg_Volume', 'Total_Volume',
                        'Avg_RSI', 'Avg_MACD', 'Avg_MACD_Signal',
                        'Avg_BB_Upper', 'Avg_BB_Lower',
                        'Avg_MA_20', 'Avg_SMA_50']

  # stats analytics
  stats_summary = {
        'market_volatility': mk_trend['Close_Price_Volatility'].mean(),
        'daily_return_trend': mk_trend['Avg_Close_Price'].pct_change().dropna(),
        'volume_linear_trend': stats.linregress(
            range(len(mk_trend)),
            mk_trend['Avg_Volume']
        ),
        'correlation_with_price': {}
    }

  # compute correlations between variables
  indicators = ['Avg_RSI', 'Avg_MACD', 'Avg_MACD_Signal',
                  'Avg_MA_20', 'Avg_SMA_50']
  
  for ind in indicators:
        stats_summary['correlation_with_price'][ind] = np.corrcoef(
            mk_trend['Avg_Close_Price'],
            mk_trend[ind]
        )[0,1]

  # compute market state: bearish, bullish or neutral
  mk_trend['Market_Condition'] = mk_trend['Avg_Close_Price'].pct_change().apply(
        lambda x: 'Bullish' if x > 0.01
        else 'Bearish' if x < -0.01
        else 'Neutral'
    )

  return {
      'market_trend': mk_trend,
      'statistics_summary': stats_summary
  }