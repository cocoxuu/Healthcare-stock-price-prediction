import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from market_analysis import add_technical_indicators, compute_market_trend
from model_training import train_lstm, prepare_training_data
from data_viz import plot_predictions
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def is_healthcare_stock(ticker):
    """
    Check if a stock belongs to the healthcare sector using yfinance.
    """
    try:
        stock = yf.Ticker(ticker) 
        info = stock.info
        sector = info.get('sector', '').lower()
        industry = info.get('industry', '').lower()
        
        # define healthcare keywords
        # this can be alteranated to other subsectors
        hc_keywords = ['health', 'pharmaceutical', 'biotech', 'medical', 'healthcare']
        for k in hc_keywords:
            if k in sector or k in industry:
                return True
        return False
    
    except Exception as e:
        print(f"Error occured :{e}")
        return False

def safe_format(value):
    """
    Safely format numpy values for display
    """
    try:
        if isinstance(value, np.ndarray):
            return float(value.item())  # convert numpy value to Python float
        return float(value)
    except:
        return value # if failed, return the original value
    
def create_dashboard():
    """
    Create a user-friendly dashboard.
    
    Note: the css and markdown format is generated by ChatGPT
    """
    # set page configuration
    st.set_page_config(layout="wide", 
                       page_icon='📈',
                       page_title="The Pulse of Profits",
                       initial_sidebar_state="expanded")
    st.title("The Pulse of Profits: Cracking Healthcare Stock Signals")
    
    # set sidebar navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Overview", "Stock Analysis", "AI Predictions"]
    )
    if page == "Overview":
        display_sector_overview()
    elif page == "Stock Analysis":
        display_stock_analysis()
    else:
        display_ai_preditions()
        
    # Footer
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9fafc;
            padding: 10px 0;
            text-align: center;
            font-size: 14px;
            color: #444444;
            border-top: 1px solid #e1e4e8;
        }
        .footer a {
            color: #0073e6;
            text-decoration: none;
            font-weight: bold;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="footer">
            Developed by <a href="mailto:kxu98274@usc.edu">Coco Xu</a> | Last updated: 12/1/2024
        </div>
        """, unsafe_allow_html=True)

def load_sector_data():
    """
    Load heathcare sector data. e.g., XLV ETF data from yahoo finance.
    If you want to observe data from other industry,
    you may try other ETFs like XLF(finance), XLK(technology) etc.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365) # set one year's data
        data = yf.download("XLV", start=start_date, end=end_date)
            
        if data.empty:
            st.error("No data found for healthcare sector ETF (XLV)")
            return None
            
        # process and clean data
        data.index = pd.to_datetime(data.index)
        data = data.reset_index()
        data = data.rename(columns={'index': 'Date'})
        data['Ticker'] = 'XLV'  
        data['Volatility'] = calculate_volatility(data)    
        processed_data = add_technical_indicators(data) # add technical indicators
        if processed_data is None:
            st.error("Failed to process technical indicators")
            return None
        return processed_data
        
    except Exception as e:
        st.error(f"Error loading sector data: {str(e)}")
        return None

def display_sector_overview():
    """
    Give a easy-to-understand, holistic view of the whole healthcare sector.
    This function is generated with the help of ChatGPT-4o.
    
    """
    st.header("Healthcare Sector Overview")
    st.markdown("""
    **This dashboard helps you understand how the overall healthcare industry is performing.**
    """)
    
    # Show sector details in a collapsible section
    with st.expander("Sector Details", expanded=True):
        st.info("""
        Data Source: **XLV (Health Care Select Sector SPDR Fund)**  
        XLV tracks healthcare stocks within the S&P 500 index. It includes pharmaceuticals, biotechnology, medical devices, and healthcare services.
        It's like checking the health of the entire healthcare industry in one glance!\n
        """)
    
    with st.spinner("Loading latest sector data..."):
        try:
            sector_data = load_sector_data()
            if sector_data is None:
                st.error("Oops! Failed to load sector data. Please try again.")
                return 
    
            # 1. market trend analysis
            market_analysis = compute_market_trend(sector_data)
            market_trend = market_analysis['market_trend']
            stats_summary = market_analysis['statistics_summary']
            
            # display key metrics in 3 column
            col1, col2, col3 = st.columns(3)
            with col1:
                market_condition = market_trend['Market_Condition'].iloc[-1]
                icon = "🟢" if market_condition == "Bullish" else "🔴" if market_condition == "Bearish" else "🟡"
                st.metric(
                    "Market Mood",
                    f"{icon} {market_condition}",
                    help="Is the market feeling optimistic or cautious?"
                )
            
            with col2:
                volatility = market_trend['Close_Price_Volatility'].iloc[-1]
                status = "High" if volatility > 20 else "Normal" if volatility > 10 else "Low"
                st.metric(
                    "Market Stability",
                    f"{status}",
                    help="How jumpy are stock prices? Lower is more stable."
                )
                
            with col3:
                rsi = market_trend['Avg_RSI'].iloc[-1]
                momentum = "Strong Buy" if rsi < 30 else "Strong Sell" if rsi > 70 else "Hold"
                st.metric(
                    "Buying Pressure",
                    momentum,
                    help="Should you consider buying or selling?"
                )
        
            # 2. Technical Analysis Dashboard 
            st.subheader("Technical Analysis")
            tabs = st.tabs(["Price Movement", "Market Signals", 'Market Relationships'])
            
            with tabs[0]:
                st.markdown("### Where are healthcare stocks heading?")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Main price line
                ax.plot(market_trend['Date'], market_trend['Avg_Close_Price'], 
                       label='Current Price', color='blue', linewidth=2)
                
                # Trend lines
                ax.plot(market_trend['Date'], market_trend['Avg_MA_20'], 
                       label='Short-term Trend', color='green', linestyle='--')
                ax.plot(market_trend['Date'], market_trend['Avg_SMA_50'], 
                       label='Long-term Trend', color='red', linestyle='--')
                
                # Normal price range
                ax.fill_between(market_trend['Date'], 
                              market_trend['Avg_BB_Upper'],
                              market_trend['Avg_BB_Lower'],
                              alpha=0.2, color='gray',
                              label='Normal Price Range')
                ax.set_title("Healthcare Sector Price Movement")
                ax.legend(loc='upper left')
                st.pyplot(fig)
                
                st.info("""
                **How to read this chart:**
                - Blue line shows current healthcare stock prices
                - Green dashed line (20-day average) shows short-term trend
                - Red dashed line (50-day average) shows long-term trend
                - Gray area shows the normal price range (Bollinger Bands)
                
                - When price moves above gray area: Potential selling opportunity
                - When price moves below gray area: Potential buying opportunity
                """)
            
            with tabs[1]:
                st.markdown("### What are the market signals telling us?")

                # Combine indicators into one meaningful interpretation
                signal_strength = (
                    "Strong Buy" if rsi < 30 and market_trend['Avg_MACD'].iloc[-1] > market_trend['Avg_MACD_Signal'].iloc[-1]
                    else "Strong Sell" if rsi > 70 and market_trend['Avg_MACD'].iloc[-1] < market_trend['Avg_MACD_Signal'].iloc[-1]
                    else "Hold"
                )
                st.metric("Overall Market Signal", signal_strength)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Buying/Selling Pressure")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(market_trend['Date'], market_trend['Avg_RSI'], color='purple')
                    
                    # create selling zone and buying zone
                    ax.fill_between(market_trend['Date'], 70, 100, 
                    color='red', alpha=0.1, label='Sell Zone')
                    ax.fill_between(market_trend['Date'], 0, 30, 
                    color='green', alpha=0.1, label='Buy Zone')
                    
                    # create selling and buying lines
                    ax.axhline(y=70, color='r', linestyle='--')
                    ax.axhline(y=30, color='g', linestyle='--')
                    ax.set_title("Market Pressure Gauge")
                    ax.legend()
                    st.pyplot(fig)
                    
                with col2:
                    st.markdown("#### Market Momentum")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(market_trend['Date'], market_trend['Avg_MACD'], label='Momentum')
                    ax.plot(market_trend['Date'], market_trend['Avg_MACD_Signal'], label='Signal Line')
                    ax.set_title("Market Momentum")
                    ax.legend()
                    st.pyplot(fig)
                
                st.info("""
                **Understanding the Signals:**
                - When the purple line enters the green zone: Consider buying
                - When the purple line enters the red zone: Consider selling
                - When lines cross in the momentum chart: Potential trend change
                """)
                
            with tabs[2]:
                st.markdown("### How do different factors affect the market?")
                
                # add a heatmap to show correlations
                correlations = pd.DataFrame.from_dict(
                    stats_summary['correlation_with_price'],
                    orient='index',
                    columns=['Impact']
            )
                    
                # visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlations[['Impact']].sort_values('Impact', ascending=False), 
                            cmap='RdBu', 
                            center=0,
                            annot=True,
                            fmt='.2f',
                            cbar_kws={'label': 'Correlation Strength'})
                ax.set_title("Market Factor Relationships")
                st.pyplot(fig)

                st.info("""
                **Understanding Market Relationships:**
                - Values close to 1: Strong positive relationship (move together)
                - Values close to -1: Strong negative relationship (move oppositely)
                - Values close to 0: Little to no relationship
                - Use these relationships to diversify your portfolio and manage risk
                """)
                
                # explain technical indicators in a expander
                st.markdown("#### Technical Indicators Explained")
                expander = st.expander("📊 What do these indicators mean?")
                with expander:
                    st.markdown("""
                    **RSI (Relative Strength Index)**
                    - Measures if a stock is overbought or oversold
                    - Range: 0-100
                    - Above 70: Potentially overbought
                    - Below 30: Potentially oversold

                    **MACD (Moving Average Convergence Divergence)**
                    - Shows momentum and trend direction
                    - When positive: Upward momentum
                    - When negative: Downward momentum

                    **Bollinger Bands**
                    - Shows normal trading range
                    - Price above bands: Possibly overvalued
                    - Price below bands: Possibly undervalued

                    **Moving Averages**
                    - 20-day: Short-term trend
                    - 50-day: Medium-term trend
                    - Crossovers can signal trend changes
                    """)
        
        except Exception as e:
            st.error(f"Error analyzing market trends: {str(e)}")
            return 

def load_stock_data(ticker):
    """
    Load individual stock data within the healthcare industry.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        with st.spinner(f"Loading data for {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No data found for ticker {ticker}")
                return None
            
            data.index = pd.to_datetime(data.index)
            data = data.reset_index()
            data = data.rename(columns={'index': 'Date'})
            data['Ticker'] = ticker
            
            return data
            
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None
    
def display_stock_analysis():
    """
    Display one ticker's information given the user input.
    """
    st.header("Single Ticker Analysis")
    st.markdown("""
    **Analyze any healthcare company's stock performance using both basic metrics and advanced AI predictions.**
    """)
    
    # select a ticker
    ticker = st.text_input("Enter a Healthcare Stock Ticker (e.g., JNJ, UNH, ABBV)",
                           placeholder="Type a valid healthcare stock ticker...").upper()
    if ticker:
        if not is_healthcare_stock(ticker):
            st.error("Oopsy! Something went wrong...\n \
                     Please enter a valid healthcare sector stock ticker.")
            return
        
        # fetch and display stock data if valid
        stock_data = load_stock_data(ticker)
        if stock_data is not None:
            stock_data = add_technical_indicators(stock_data)

            # 1. quick overview of the input ticker
            st.subheader(f"Quick Overview: {ticker}")
            col1, col2, col3 = st.columns(3)              
            with col1:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                price_change = ((current_price - prev_price) / prev_price) * 100
                st.metric("Current Price", 
                         f"${current_price:.2f}", 
                         f"{price_change:+.2f}%",
                         help="Today's closing price vs yesterday")
            
            with col2:
                    rsi = stock_data['RSI'].iloc[-1]
                    rsi_status = (
                        "Overbought" if rsi > 70
                        else "Oversold" if rsi < 30
                        else "Neutral"
                    )
                    st.metric(
                        "Momentum (RSI)",
                        f"{rsi:.1f}",
                        rsi_status,
                        help="RSI > 70: Consider Selling, RSI < 30: Consider Buying"
                    )
                       
            with col3:
                    volatility = calculate_volatility(stock_data).iloc[-1]
                    vol_status = (
                        "High" if volatility > 25
                        else "Medium" if volatility > 15
                        else "Low"
                    )
                    st.metric(
                        "Price Stability",
                        f"{vol_status}",
                        f"{volatility:.1f}% volatility",
                        help="Based on price movements over the last 20 days"
                    )   
            
            # 2. dataviz of the input ticker
            st.subheader("Technical Analysis")
            display_technical_charts(stock_data, ticker)
            
        else:
            st.error("Failed to load stock data. Please try again.")

def display_technical_charts(data, ticker):
    """
    Display a simplified and more intuitive technical chart.
    """
    try:
        # create figure with secondary y-axis
        fig = plt.figure(figsize=(12,8))
        
        # create 2 subplots
        gs = fig.add_gridspec(2,1,height_ratios=[2,1], hspace=0.5)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # price chart(top)
        ax1.plot(data['Date'], data['Close'], label='Close Price', color='#2962FF', linewidth=2)
        if 'MA_20' in data.columns:
            ax1.plot(data['Date'], data['MA_20'], 
                     label='20-day Trend', 
                    color='#FF6D00',
                    alpha=0.7,
                    linestyle='--')
        
        # add Bollinger Bands 
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            ax1.fill_between(data['Date'], 
                           data['BB_Upper'], 
                           data['BB_Lower'], 
                           alpha=0.1, 
                           color='gray',
                           label='Normal Range')
            
        # customize price chart
        ax1.set_title(f"{ticker} Technical Analysis", pad=20, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Prie ($)", fontsize=10)

        
        # RSI chart (Bottom)
        if 'RSI' in data.columns:
            ax2.plot(data['Date'], data['RSI'], color='#673AB7', linewidth=1.5, label='Momentum')
            
            # add zones
            ax2.fill_between(data['Date'], 70, 100, alpha=0.1, color='red', label='Overbought')
            ax2.fill_between(data['Date'], 0, 30, alpha=0.1, color='green', label='Oversold')
            ax2.axhline(y=70, color='red', linestyle=':', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle=':', alpha=0.5)
            
            # customize RSI chart
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
        # Rotate x-axis dates for better readability
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        # ax1.set_xticklabels([])  # remove the top chart X-axis label
        ax2.set_xlabel('Date', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)

        # add simplified explanation
        st.info("""
            **🎓 How to read this chart**
            
            **Price Chart (Top)**
            - Blue line: Current stock price
            - Orange dashed line: 20-day trend line
            - Gray area: Normal trading range
            
            **Momentum Gauge (Bottom)**
            - Purple line: Shows buying/selling pressure (RSI)
            - Red zone (above 70): Strong selling pressure
            - Green zone (below 30): Strong buying pressure
            
            **Trading Signals**
            1. When price crosses above/below trend line = Potential trend change
            2. When price moves outside gray area = Unusual price movement
            3. When momentum enters red/green zones = Potential reversal points
            """)
        
    except Exception as e:
        st.error(f"Error displaying technical chart: {str(e)}")
        st.write("Available columns:", data.columns.tolist())

        
def display_ai_preditions():
    """
    Display AI-driven stock market predictions with a concise and user-friendly interface.
    """
    st.header("AI-Powered Market Insights")
    st.markdown("**Upload historical stock data to generate AI-driven predictions and insights.**")
    
    # an explanation of CSV format
    with st.expander("📄 CSV Format Requirements"):
        st.markdown("""
        **Required columns:**
        - `Date`: (YYYY-MM-DD format)
        - `Open`, `High`, `Low`, `Close`, `Volume`: (Numeric values)

        **Optional column:**
        - `Ticker`: (For datasets with multiple stocks)

        Our AI model automatically calculates technical indicators and predicts future trends.
        """)

    uploaded_file = st.file_uploader("Upload CSV file with historical data", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) 
        
        with st.spinner("AI is analyzing market patterns..."):
            df = add_technical_indicators(df)
            features, target, target_scaled, target_scaler = prepare_training_data(df)
            results = train_lstm(features, target_scaled, target_scaler)

            if results is None:
                st.error("Failed to train model, please check your data format.")
                return None
            
            # extract key metrics
            latest_actual = float(results['true_values'][-1])
            latest_pred = float(results['predictions'][-1].item())
            price_change_pred = ((latest_pred - latest_actual) / latest_actual) * 100
            accuracy = results['metrics']['r2']*100
            error_margin = results['metrics']['rmse']
            
            # display key metrics
            st.subheader("📊 Market Prediction Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Prediction Confidence",
                    f"{accuracy:.1f}%",
                    help="Based on historical prediction accuracy (R² score)"
                )
                st.metric(
                    "Average Error Margin",
                    f"±${safe_format(error_margin):.2f}",
                    help="Average difference between predicted and actual prices (RMSE)"
                )
            with col2:
                st.metric(
                    "Next 5-day Period Prediction",
                    f"${safe_format(latest_pred):.2f}",
                    f"{safe_format(price_change_pred):+.1f}%",
                    delta_color="normal"   
                )
                st.metric(
                    "Current Price", # add current price as a reference
                    f"${safe_format(latest_actual):.2f}"
                )
                
            #2. visualization 
            st.subheader("🎯 Price Predictions VS Actual")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_predictions(results['true_values'], results['predictions'], ax=ax)
            st.pyplot(fig)
            
            # 3. summary and tradiing recommendations
            display_analysis_results(price_change_pred, accuracy,error_margin,
                                     latest_actual, results)

def display_analysis_results(price_change_pred, accuracy, error_margin, latest_actual, results):
    """
    Display user-friendly market analysis results with clear explanations
    and visual indicators.
    
    This function is generated with the help of ChatGPT-4o.
    """
    
    # Market Direction Analysis
    trend_strength = abs(safe_format(price_change_pred))
    sentiment_level = "Strong " if trend_strength > 7 else "Moderate " if trend_strength > 3 else "Mild "
    sentiment_direction = "Upward Trend" if safe_format(price_change_pred) > 0 else "Downward Trend"
    sentiment = f"{sentiment_level}{sentiment_direction}"
    
    # fefine market status with clear indicators
    def get_market_status(accuracy, price_change):
        if accuracy < 60:
            return "Uncertain", "yellow"
        if price_change > 7:
            return "Strongly Bullish", "green"
        if price_change > 3:
            return "Bullish", "green"
        if price_change < -7:
            return "Strongly Bearish", "red"
        if price_change < -3:
            return "Bearish", "red"
        return "Neutral", "blue"
    
    market_status, status_color = get_market_status(safe_format(accuracy), safe_format(price_change_pred))
    
    # overview Section with clear visual indicators
    st.subheader("🎯 Overview of Your Input Stocks")
    cols = st.columns(3)
    
    with cols[0]:
        st.metric(
            "Predicted Price Movement",
            f"${safe_format(latest_actual):.2f} → ${safe_format(latest_actual * (1 + price_change_pred/100)):.2f}",
            f"{safe_format(price_change_pred):+.1f}%",
            delta_color="normal"
        )     
    with cols[1]:
        st.metric(
            "Market Sentiment",
            market_status,
            help="Overall market direction based on AI analysis"
        )   
    with cols[2]:
        st.metric(
            "Prediction Confidence",
            f"{safe_format(accuracy):.1f}%",
            help="How confident our AI is in this prediction"
        )

    # detailed Analysis Section
    st.subheader("🔍 Detailed Analysis")
    
    # risk assesment
    risk_ratio = safe_format(error_margin)/safe_format(latest_actual)
    risk_level = (
        ("🔴 High Risk", "Higher market volatility detected")
        if risk_ratio > 0.1
        else ("🟡 Moderate Risk", "Normal market fluctuations expected")
        if risk_ratio > 0.05
        else ("🟢 Low Risk", "Market showing stability")
    )
    
    st.info(f"**Risk Level:** {risk_level[0]}\n\n{risk_level[1]}")

    # trading Insights
    st.subheader("💡 Trading Insights")
    
    if safe_format(accuracy) < 60:
        st.warning("""
        **Market signals are currently unclear**
        - Consider waiting for more definitive trends
        - Monitor market for clearer signals
        - Focus on risk management
        """)
        
    else:
        # I add safe_format here to avoid bugs
        if safe_format(price_change_pred) > 5:
            st.success("""
            **Positive Market Outlook**
            - Strong buying signals detected
            - Market momentum appears positive
            - Consider gradual position building
            """)
        elif safe_format(price_change_pred) < -5:
            st.error("""
            **Cautious Market Outlook**
            - Defensive positioning recommended
            - Consider profit taking if holding
            - Watch for market stabilization
            """)
        else:
            st.info("""
            **Neutral Market Conditions**
            - Market showing stability
            - Maintain balanced positions
            - Watch for trend development
            """)
    
    # technical indicator explanations
    with st.expander("📈 View Statistical Metrics"):
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MSE'],
            'Value': [
                f"{safe_format(results['metrics']['r2']):.4f}",
                f"${safe_format(results['metrics']['rmse']):.2f}",
                f"${safe_format(results['metrics']['mse']):.2f}"
            ],
            'Explanation': [
                "How well our predictions match actual market movements",
                "Average prediction error in dollars",
                "Prediction consistency measure"
            ]
        })
        st.table(metrics_df)

    with st.expander("❓ Understanding This Analysis"):
        st.markdown("""
        
        1. **Market Overview** shows the predicted price movement and our confidence level
        2. **Risk Level** indicates current market volatility and uncertainty
        3. **Trading Insights** provide actionable suggestions based on the analysis
        4. **Technical Details** show the underlying metrics for advanced users
        
        Remember: This analysis is one of many tools for making investment decisions. 
        Always consider multiple factors and consult financial professionals when needed.
        """)
        
def calculate_volatility(data):
    """
    Calculate rolling volatility using log returns.
    Returns volatility as a percentage.
    """
    try:
        returns = np.log(data['Close'] / data['Close'].shift(1)) 
        volatility = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100 
        volatility.fillna(
            returns.expanding().std() * np.sqrt(252) * 100,
            inplace=True
        )
        
        return volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return pd.Series([np.nan] *len(data))

if __name__ == '__main__':
    create_dashboard()
