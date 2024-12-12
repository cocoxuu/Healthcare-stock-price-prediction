import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_tech_indicators(df, stock_name='Stock'):
    """
    Visualize technical indicators: Close, MAs, Volume and RSI.
    """
    # turn date into datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(14, 10))
    
    # plot close price with MAs
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['MA_20'], label='MA 20', color='orange', linestyle='--')
    plt.plot(df['Date'], df['SMA_50'], label='SMA 50', color='green', linestyle='--')
    plt.title(f'{stock_name} Close Price and Moving Averages')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # set date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # set by month
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  
    plt.gcf().autofmt_xdate()  
    
    # plot volume
    plt.subplot(3, 1, 2)
    plt.bar(df['Date'], df['Volume'], color='grey')
    plt.title(f'{stock_name} Volume')
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid(True, axis='y', alpha=0.3)

    # plot RSI
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', label = "Overbought (70)", linestyle='--')
    plt.axhline(30, color='green', label = "Oversold (30)",linestyle='--')
    plt.title(f'{stock_name} RSI')
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def plot_market_trend(df):
    """
    Plot the overall market trend using mean values of specified indicators.
    """
    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize=(14, 7))
    indicators = {
        'Avg_Close_Price': {'color': 'blue', 'label': 'Average Close Price'},
        'Avg_MA_20': {'color': 'orange', 'label': 'MA 20', 'linestyle': '--'},
        'Avg_SMA_50': {'color': 'green', 'label': 'SMA 50', 'linestyle': '--'},
        'Avg_BB_Upper': {'color': 'red', 'label': 'Bollinger Upper', 'linestyle': ':'},
        'Avg_BB_Lower': {'color': 'red', 'label': 'Bollinger Lower', 'linestyle': ':'}
    }

    for indicator, style in indicators.items():
        if indicator in df.columns:
            plt.plot(df['Date'], df[indicator], 
                    label=style['label'], 
                    color=style['color'],
                    linestyle=style.get('linestyle', '-'))
    
    plt.title('Overall Market Trend (Mean values over time)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # set date format %Y-%m
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    
def plot_sector_daily_return(df):
    """
    Plot the daily returns for the sector.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sector_Daily_Return'] = df['Avg_Close_Price'].pct_change() * 100 # use close_mean to compute

    plt.figure(figsize=(14, 7))

    # sector daily return
    plt.plot(df['Date'], df['Sector_Daily_Return'], 
             label='Sector Daily Return', color='purple', alpha=0.3)
    
    # 30-day SMA
    plt.plot(df['Date'], df['Sector_Daily_Return'].rolling(window=30).mean(), 
             label='30-Day SMA', color='orange')
    
    plt.title('Healthcare Sector Average Daily Return (Smoothed)')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # set date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_market_indicators(df):
    """
    Plot technical indicators: RSI, MACD
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # plot RSI
    ax1.plot(df['Date'], df['Avg_RSI'], label='RSI', color='purple')
    ax1.axhline(70, color='red', label="Overbought (70)", linestyle='--')
    ax1.axhline(30, color='green', label="Oversold (30)", linestyle='--')
    ax1.set_title('Market Average RSI')
    ax1.set_ylabel('RSI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # plot MACD
    ax2.plot(df['Date'], df['Avg_MACD'], label='MACD', color='blue')
    ax2.plot(df['Date'], df['Avg_MACD_Signal'], label='Signal Line', color='red')
    ax2.set_title('Market Average MACD')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # set dateime
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show() 

def plot_training_history(history, ax=None):
    """
    Plot the training and validation loss of the model.
    This function is generated with the help of ChatGPT-4o
    
    """
    if ax is None:
        _, ax = plt.subplots() # 
    try:
        history_dict = history.history if hasattr(history, 'history') else history

        ax.plot(history_dict['loss'], label='Training Loss')
        if 'val_loss' in history_dict:
            ax.plot(history_dict['val_loss'], label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend(loc='upper right')
        
    except AttributeError:
        raise ValueError("Invalid history object provided")
    
    return ax


def plot_predictions(true_values, predictions, ax=None):
    """
    Create an improved visualization of price predictions vs actual values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    time_index = range(len(true_values)) # use actual data points
    
    # styling
    ax.plot(time_index, true_values, 'o', color='#4A90E2', alpha=0.5, 
            markersize=4, label='Actual Prices')
    ax.plot(time_index, predictions, color='#E53935', linewidth=2, 
            label='Predicted Prices')
    
    # set title, lable, legend and spine
    ax.set_title('Price Predictions vs Actual Values', 
                fontsize=14, pad=20)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95,
             shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False) # remove top and right spines
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return ax
