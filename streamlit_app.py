import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Algorithmic Trading Analysis: MACD Strategy with SMA and EMA")
st.markdown("""
Welcome! This application was created by Dymasius Yusuf Sitepu (G2303593E) presents an analysis of MACD trading strategies using both Simple Moving Averages (SMA) 
and Exponential Moving Averages (EMA) on four different assets. The analysis includes comparing trading 
profits across assets and evaluating different parameter sets to determine optimal trading configurations.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Introduction", "Data Overview", "Part 1: SMA Strategy", "Part 2: EMA Strategy", "Comparison & Conclusion"]
)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load from the current directory first
        data = pd.read_csv('Data Daily Asset Prices v2.csv')
    except FileNotFoundError:
        # If not found, try the full path
        data = pd.read_csv('/home/ubuntu/upload/Data Daily Asset Prices v2.csv')
    return data

data = load_data()

# Define functions for SMA strategy
def calculate_sma(prices, window):
    """Calculate Simple Moving Average for a given window size"""
    return prices.rolling(window=window).mean()

def calculate_trading_signal(ma_short, ma_long, delta=0.005):
    """Calculate trading signal based on MA crossover strategy"""
    # Calculate relative difference
    relative_diff = (ma_short - ma_long) / ma_long
    
    # Generate buy/sell signals
    buy_signal = relative_diff > delta
    sell_signal = relative_diff < -delta
    
    return buy_signal, sell_signal, relative_diff

def backtest_strategy(prices, short_window, long_window, delta=0.005, initial_cash=100, strategy_type="SMA"):
    """
    Backtest trading strategy (SMA or EMA)
    
    Parameters:
    -----------
    prices : pandas Series
        Asset prices
    short_window : int
        Short-term MA window
    long_window : int
        Long-term MA window
    delta : float
        Threshold for trading signal
    initial_cash : float
        Initial cash amount
    strategy_type : str
        "SMA" or "EMA"
        
    Returns:
    --------
    portfolio_df : pandas DataFrame
        Daily portfolio values and related data
    trades_df : pandas DataFrame
        Record of all trades
    final_portfolio_value : float
        Final portfolio value
    """
    # Calculate MAs based on strategy type
    if strategy_type == "SMA":
        ma_short = calculate_sma(prices, short_window)
        ma_long = calculate_sma(prices, long_window)
    else:  # EMA
        beta_short = 2 / (short_window + 1)
        beta_long = 2 / (long_window + 1)
        ma_short = prices.ewm(span=short_window, adjust=False).mean()
        ma_long = prices.ewm(span=long_window, adjust=False).mean()
    
    # Calculate trading signals
    buy_signal, sell_signal, relative_diff = calculate_trading_signal(ma_short, ma_long, delta)
    
    # Initialize portfolio tracking variables
    cash = initial_cash
    position = 0  # Number of shares held
    portfolio_values = []  # Track daily portfolio value
    trades = []  # Track trades
    
    # Loop through each day (starting after we have both MAs)
    for i in range(max(short_window, long_window), len(prices)):
        date = i  # Using index as date for simplicity
        current_price = prices.iloc[i]
        
        # Record current portfolio value
        if position == 0:
            portfolio_value = cash
        else:
            portfolio_value = position * current_price
        
        portfolio_values.append({
            'day': date,
            'price': current_price,
            'cash': cash,
            'position': position,
            'portfolio_value': portfolio_value,
            'ma_short': ma_short.iloc[i],
            'ma_long': ma_long.iloc[i],
            'relative_diff': relative_diff.iloc[i]
        })
        
        # Check for sell signal when we have a position
        if position > 0 and sell_signal.iloc[i]:
            cash = position * current_price
            trades.append({
                'day': date,
                'action': 'SELL',
                'price': current_price,
                'quantity': position,
                'value': cash
            })
            position = 0
        
        # Check for buy signal when we have cash
        elif position == 0 and buy_signal.iloc[i]:
            position = cash / current_price
            trades.append({
                'day': date,
                'action': 'BUY',
                'price': current_price,
                'quantity': position,
                'value': cash
            })
            cash = 0
    
    # Convert to DataFrames
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['day', 'action', 'price', 'quantity', 'value'])
    
    # Calculate final portfolio value (if still holding position)
    final_portfolio_value = portfolio_df['portfolio_value'].iloc[-1]
    
    return portfolio_df, trades_df, final_portfolio_value

def run_strategy_all_assets(data, short_window, long_window, delta=0.005, initial_cash=100, strategy_type="SMA"):
    """Run strategy on all assets and compare results"""
    results = {}
    
    # Run strategy for each asset
    for asset in ['Asset1', 'Asset2', 'Asset3', 'Asset4']:
        portfolio_df, trades_df, final_value = backtest_strategy(
            data[asset], short_window, long_window, delta, initial_cash, strategy_type
        )
        
        results[asset] = {
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'final_value': final_value,
            'return_pct': (final_value - initial_cash) / initial_cash * 100
        }
    
    return results

def run_strategy_parameter_comparison(data, asset, parameter_sets, initial_cash=100, strategy_type="SMA"):
    """Compare different parameter sets for a specific asset"""
    results = {}
    
    for params in parameter_sets:
        short_window, long_window, delta = params
        param_key = f"Set {chr(97 + len(results))}: S={short_window}, L={long_window}, δ={delta}"
        
        portfolio_df, trades_df, final_value = backtest_strategy(
            data[asset], short_window, long_window, delta, initial_cash, strategy_type
        )
        
        results[param_key] = {
            'portfolio_df': portfolio_df,
            'trades_df': trades_df,
            'final_value': final_value,
            'return_pct': (final_value - initial_cash) / initial_cash * 100,
            'params': params
        }
    
    return results

# Introduction page
if page == "Introduction":
    st.header("Introduction to MACD Trading Strategies")
    
    st.markdown("""
    ## What is MACD?
    
    Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship 
    between two moving averages of a security's price. The MACD is calculated by subtracting the long-term moving 
    average from the short-term moving average.
    
    ## Trading Strategies Implemented
    
    In this analysis, we implement and evaluate two types of MACD trading strategies:
    
    1. **Part 1**: MACD using Simple Moving Averages (SMA)
    2. **Part 2**: MACD using Exponential Moving Averages (EMA)
    
    ## Trading Rules
    
    The trading rules for both strategies are as follows:
    
    - Start with $100 cash
    - Buy signal: When the relative difference between short and long moving averages exceeds a threshold (δ)
    - Sell signal: When the relative difference falls below the negative threshold (-δ)
    - All cash must be used when buying (no partial buys)
    - All positions must be sold when selling (no partial sells)
    - Short selling is not allowed
    - If a position is open on the final day, the portfolio value is calculated as the number of units held multiplied by the closing price
    
    ## Parameters
    
    For both strategies, we test the following parameter sets:
    
    - Set (a): S=12, L=26, δ=0.005 (default)
    - Set (b): S=5, L=10, δ=0.005
    - Set (c): S=10, L=20, δ=0.005
    
    Where:
    - S is the short-term moving average window
    - L is the long-term moving average window
    - δ is the threshold for generating trading signals
    """)
    
    st.info("Navigate through the sidebar to explore the data, analysis, and results.")

# Data Overview page
elif page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("First Few Rows of the Dataset")
    st.dataframe(data.head())
    
    st.subheader("Basic Statistics")
    st.dataframe(data.describe())
    
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    
    st.subheader("Price Movements of All Assets")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot each asset
    for i, asset in enumerate(['Asset1', 'Asset2', 'Asset3', 'Asset4']):
        axes[i].plot(data['DAY'], data[asset])
        axes[i].set_title(f'{asset} Price Movement')
        axes[i].set_xlabel('Day')
        axes[i].set_ylabel('Price')
        axes[i].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Correlation Between Assets")
    correlation = data[['Asset1', 'Asset2', 'Asset3', 'Asset4']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Between Assets')
    st.pyplot(fig)

# Part 1: SMA Strategy page
elif page == "Part 1: SMA Strategy":
    st.header("Part 1: MACD Using SMA")
    
    st.markdown("""
    In this section, we implement the MACD trading strategy using Simple Moving Averages (SMA). 
    The SMA is calculated as the arithmetic mean of prices over a specified period.
    """)
    
    # Question 1: SMA Strategy for All Assets
    st.subheader("Question 1: SMA Strategy for All Assets")
    
    # Parameters for SMA strategy
    st.markdown("#### Default Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Short Window (S)", "12 days")
    with col2:
        st.metric("Long Window (L)", "26 days")
    with col3:
        st.metric("Threshold (δ)", "0.005")
    
    # Run SMA strategy for all assets
    sma_results = run_strategy_all_assets(data, short_window=12, long_window=26, delta=0.005, strategy_type="SMA")
    
    # Display results in a table
    st.markdown("#### Results")
    sma_summary = {
        'Asset': list(sma_results.keys()),
        'Final Value ($)': [f"${result['final_value']:.2f}" for result in sma_results.values()],
        'Return (%)': [f"{result['return_pct']:.2f}%" for result in sma_results.values()],
        'Number of Trades': [len(result['trades_df']) for result in sma_results.values()]
    }
    st.table(pd.DataFrame(sma_summary))
    
    # Plot portfolio performance
    st.markdown("#### Portfolio Performance Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for asset, result in sma_results.items():
        portfolio_df = result['portfolio_df']
        ax.plot(portfolio_df['day'], portfolio_df['portfolio_value'], label=asset)
    
    ax.set_title('SMA Strategy Portfolio Performance Comparison')
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Answer to Question 1
    st.markdown("#### Answer to Question 1")
    st.markdown("""
    Based on the SMA trading strategy with parameters S=12, L=26, and δ=0.005, the ranking of assets by trading profits is:
    
    1. **Asset1 (45.19%)** - Significantly outperformed the other assets
    2. **Asset3 (4.10%)**
    3. **Asset4 (3.79%)**
    4. **Asset2 (0.87%)** - Barely broke even
    
    Asset1 generated a return of 45.19% over the trading period, which is more than 10 times higher than the next best performing asset. The number of trades executed was similar across assets (5-7 trades), suggesting that the difference in performance was not due to trading frequency but rather to the price movement patterns of each asset and how well they aligned with the SMA strategy parameters.
    """)
    
    # Question 2: Parameter Comparison for Asset1
    st.subheader("Question 2: Parameter Comparison for Asset1")
    
    # Define parameter sets
    parameter_sets = [
        (12, 26, 0.005),  # Set (a)
        (5, 10, 0.005),   # Set (b)
        (10, 20, 0.005)   # Set (c)
    ]
    
    # Run parameter comparison
    param_results = run_strategy_parameter_comparison(data, 'Asset1', parameter_sets, strategy_type="SMA")
    
    # Display results in a table
    st.markdown("#### Results")
    param_summary = {
        'Parameter Set': list(param_results.keys()),
        'Final Value ($)': [f"${result['final_value']:.2f}" for result in param_results.values()],
        'Return (%)': [f"{result['return_pct']:.2f}%" for result in param_results.values()],
        'Number of Trades': [len(result['trades_df']) for result in param_results.values()]
    }
    st.table(pd.DataFrame(param_summary))
    
    # Plot parameter comparison
    st.markdown("#### Parameter Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, result in param_results.items():
        portfolio_df = result['portfolio_df']
        ax.plot(portfolio_df['day'], portfolio_df['portfolio_value'], label=label)
    
    ax.set_title('Asset1 SMA Strategy Parameter Comparison')
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Trading signals visualization
    st.markdown("#### Trading Signals Visualization")
    
    # Select parameter set for visualization
    selected_param_set = st.selectbox(
        "Select parameter set to visualize trading signals:",
        list(param_results.keys())
    )
    
    # Get selected parameter set data
    selected_result = param_results[selected_param_set]
    portfolio_df = selected_result['portfolio_df']
    short_window, long_window, delta = selected_result['params']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot asset price and MAs
    ax1.plot(portfolio_df['day'], portfolio_df['price'], label='Asset1 Price')
    ax1.plot(portfolio_df['day'], portfolio_df['ma_short'], label=f'SMA({short_window})')
    ax1.plot(portfolio_df['day'], portfolio_df['ma_long'], label=f'SMA({long_window})')
    ax1.set_title(f'Asset1 Price and SMAs - {selected_param_set}')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot relative difference and thresholds
    ax2.plot(portfolio_df['day'], portfolio_df['relative_diff'], label='Relative Difference')
    ax2.axhline(y=delta, color='g', linestyle='--', label=f'Buy Threshold ({delta})')
    ax2.axhline(y=-delta, color='r', linestyle='--', label=f'Sell Threshold (-{delta})')
    ax2.set_title(f'SMA Relative Difference - {selected_param_set}')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Relative Difference')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Answer to Question 2
    st.markdown("#### Answer to Question 2")
    st.markdown("""
    For Asset1, we tested three different parameter sets:
    
    - **Set (a)**: S=12, L=26, δ=0.005 - Return: 45.19%, Trades: 7
    - **Set (b)**: S=5, L=10, δ=0.005 - Return: 50.99%, Trades: 15
    - **Set (c)**: S=10, L=20, δ=0.005 - Return: 74.46%, Trades: 7
    
    **Best Parameter Set**: Set (c) with S=10, L=20, δ=0.005 is the best parameter set for Asset1, yielding a 74.46% return.
    
    **Ranking of Parameter Sets by Trading Profits**:
    1. Set (c): S=10, L=20, δ=0.005 (74.46%)
    2. Set (b): S=5, L=10, δ=0.005 (50.99%)
    3. Set (a): S=12, L=26, δ=0.005 (45.19%)
    
    **Analysis**:
    - Set (c) provided the highest return while maintaining the same number of trades (7) as the default parameter set (a).
    - Set (b) had more than twice as many trades (15) compared to the other sets, indicating higher sensitivity to price movements due to the shorter windows. While this resulted in better performance than set (a), it didn't match the efficiency of set (c).
    - The intermediate window lengths in set (c) appear to strike an optimal balance for Asset1's price patterns, capturing meaningful trends while avoiding excessive trading on short-term noise.
    - All parameter sets were profitable, suggesting that the SMA strategy is generally effective for Asset1 across different timeframes.
    
    This analysis demonstrates the importance of parameter optimization in trading strategies. The default parameters (set a) performed well, but customizing the window lengths to better match the asset's price dynamics resulted in significantly improved returns.
    """)

# Part 2: EMA Strategy page
elif page == "Part 2: EMA Strategy":
    st.header("Part 2: MACD Using EMA")
    
    st.markdown("""
    In this section, we implement the MACD trading strategy using Exponential Moving Averages (EMA). 
    The EMA gives more weight to recent prices, making it more responsive to new information compared to SMA.
    """)
    
    # Question 3: EMA Strategy for All Assets
    st.subheader("Question 3: EMA Strategy for All Assets")
    
    # Parameters for EMA strategy
    st.markdown("#### Default Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Short Window (S)", "12 days")
    with col2:
        st.metric("Long Window (L)", "26 days")
    with col3:
        st.metric("Threshold (δ)", "0.005")
    
    # Run EMA strategy for all assets
    ema_results = run_strategy_all_assets(data, short_window=12, long_window=26, delta=0.005, strategy_type="EMA")
    
    # Display results in a table
    st.markdown("#### Results")
    ema_summary = {
        'Asset': list(ema_results.keys()),
        'Final Value ($)': [f"${result['final_value']:.2f}" for result in ema_results.values()],
        'Return (%)': [f"{result['return_pct']:.2f}%" for result in ema_results.values()],
        'Number of Trades': [len(result['trades_df']) for result in ema_results.values()]
    }
    st.table(pd.DataFrame(ema_summary))
    
    # Plot portfolio performance
    st.markdown("#### Portfolio Performance Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for asset, result in ema_results.items():
        portfolio_df = result['portfolio_df']
        ax.plot(portfolio_df['day'], portfolio_df['portfolio_value'], label=asset)
    
    ax.set_title('EMA Strategy Portfolio Performance Comparison')
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Answer to Question 3
    st.markdown("#### Answer to Question 3")
    st.markdown("""
    Based on the EMA trading strategy with parameters S=12, L=26, and δ=0.005, the ranking of assets by trading profits is:
    
    1. **Asset1 (29.58%)** - Outperformed the other assets
    2. **Asset4 (7.32%)**
    3. **Asset3 (3.03%)**
    4. **Asset2 (2.47%)**
    
    Asset1 again outperformed the other assets, generating a return of 29.58% over the trading period. However, this return is lower than what was achieved with the SMA strategy (45.19%). Interestingly, Asset4 performed better with the EMA strategy (7.32%) than with the SMA strategy (3.79%), moving it to second place in the ranking.
    
    The number of trades executed varied slightly across assets (3-7 trades), with Asset3 having the fewest trades (3). This suggests that the EMA strategy generated fewer trading signals for Asset3 compared to the SMA strategy, which had 5 trades.
    """)
    
    # Question 4: Parameter Comparison for Asset1 using EMA
    st.subheader("Question 4: Parameter Comparison for Asset1 using EMA")
    
    # Define parameter sets
    parameter_sets = [
        (12, 26, 0.005),  # Set (a)
        (5, 10, 0.005),   # Set (b)
        (10, 20, 0.005)   # Set (c)
    ]
    
    # Run parameter comparison
    ema_param_results = run_strategy_parameter_comparison(data, 'Asset1', parameter_sets, strategy_type="EMA")
    
    # Display results in a table
    st.markdown("#### Results")
    ema_param_summary = {
        'Parameter Set': list(ema_param_results.keys()),
        'Final Value ($)': [f"${result['final_value']:.2f}" for result in ema_param_results.values()],
        'Return (%)': [f"{result['return_pct']:.2f}%" for result in ema_param_results.values()],
        'Number of Trades': [len(result['trades_df']) for result in ema_param_results.values()]
    }
    st.table(pd.DataFrame(ema_param_summary))
    
    # Plot parameter comparison
    st.markdown("#### Parameter Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, result in ema_param_results.items():
        portfolio_df = result['portfolio_df']
        ax.plot(portfolio_df['day'], portfolio_df['portfolio_value'], label=label)
    
    ax.set_title('Asset1 EMA Strategy Parameter Comparison')
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Trading signals visualization
    st.markdown("#### Trading Signals Visualization")
    
    # Select parameter set for visualization
    selected_ema_param_set = st.selectbox(
        "Select parameter set to visualize trading signals:",
        list(ema_param_results.keys())
    )
    
    # Get selected parameter set data
    selected_ema_result = ema_param_results[selected_ema_param_set]
    ema_portfolio_df = selected_ema_result['portfolio_df']
    ema_short_window, ema_long_window, ema_delta = selected_ema_result['params']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot asset price and MAs
    ax1.plot(ema_portfolio_df['day'], ema_portfolio_df['price'], label='Asset1 Price')
    ax1.plot(ema_portfolio_df['day'], ema_portfolio_df['ma_short'], label=f'EMA({ema_short_window})')
    ax1.plot(ema_portfolio_df['day'], ema_portfolio_df['ma_long'], label=f'EMA({ema_long_window})')
    ax1.set_title(f'Asset1 Price and EMAs - {selected_ema_param_set}')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot relative difference and thresholds
    ax2.plot(ema_portfolio_df['day'], ema_portfolio_df['relative_diff'], label='Relative Difference')
    ax2.axhline(y=ema_delta, color='g', linestyle='--', label=f'Buy Threshold ({ema_delta})')
    ax2.axhline(y=-ema_delta, color='r', linestyle='--', label=f'Sell Threshold (-{ema_delta})')
    ax2.set_title(f'EMA Relative Difference - {selected_ema_param_set}')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Relative Difference')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Answer to Question 4
    st.markdown("#### Answer to Question 4")
    st.markdown("""
    For Asset1, we tested three different parameter sets using the EMA strategy:
    
    - **Set (a)**: S=12, L=26, δ=0.005 - Return: 29.58%, Trades: 7
    - **Set (b)**: S=5, L=10, δ=0.005 - Return: 52.37%, Trades: 11
    - **Set (c)**: S=10, L=20, δ=0.005 - Return: 35.18%, Trades: 7
    
    **Best Parameter Set**: Set (b) with S=5, L=10, δ=0.005 is the best parameter set for Asset1 using the EMA strategy, yielding a 52.37% return.
    
    **Ranking of Parameter Sets by Trading Profits**:
    1. Set (b): S=5, L=10, δ=0.005 (52.37%)
    2. Set (c): S=10, L=20, δ=0.005 (35.18%)
    3. Set (a): S=12, L=26, δ=0.005 (29.58%)
    
    **Analysis**:
    - Set (b) provided the highest return but required more trades (11) compared to the other sets (7 each). This suggests that the shorter windows in set (b) made the strategy more responsive to price changes, resulting in more frequent trading.
    - Set (c) performed better than the default parameters (set a), indicating that intermediate window lengths are more suitable for Asset1 when using the EMA strategy.
    - All parameter sets were profitable, but the returns were generally lower than those achieved with the SMA strategy, except for set (b) which performed better with EMA (52.37%) than with SMA (50.99%).
    
    **Comparison with SMA Strategy**:
    - For Asset1, the SMA strategy with set (c) parameters (S=10, L=20) yielded the highest overall return (74.46%), outperforming the best EMA parameter set (b) which returned 52.37%.
    - The EMA strategy generally resulted in fewer trades for most assets compared to the SMA strategy, suggesting that EMA signals are less sensitive to short-term price fluctuations.
    - The ranking of assets was similar between the two strategies, with Asset1 consistently outperforming the others, but the EMA strategy showed better performance for Asset4.
    
    This analysis demonstrates that while both SMA and EMA strategies can be effective, the SMA strategy appears to be more suitable for Asset1 when optimized with the right parameters. However, the EMA strategy might be preferable for certain assets like Asset4, or in situations where fewer trades are desired to minimize transaction costs.
    """)

# Comparison & Conclusion page
elif page == "Comparison & Conclusion":
    st.header("Comparison & Conclusion")
    
    # Comparison of SMA and EMA strategies
    st.subheader("Comparison of SMA and EMA Strategies")
    
    # Run both strategies for all assets
    sma_results = run_strategy_all_assets(data, short_window=12, long_window=26, delta=0.005, strategy_type="SMA")
    ema_results = run_strategy_all_assets(data, short_window=12, long_window=26, delta=0.005, strategy_type="EMA")
    
    # Create comparison table
    comparison_data = []
    for asset in ['Asset1', 'Asset2', 'Asset3', 'Asset4']:
        comparison_data.append({
            'Asset': asset,
            'SMA Final Value': f"${sma_results[asset]['final_value']:.2f}",
            'SMA Return (%)': f"{sma_results[asset]['return_pct']:.2f}%",
            'SMA Trades': len(sma_results[asset]['trades_df']),
            'EMA Final Value': f"${ema_results[asset]['final_value']:.2f}",
            'EMA Return (%)': f"{ema_results[asset]['return_pct']:.2f}%",
            'EMA Trades': len(ema_results[asset]['trades_df'])
        })
    
    st.table(pd.DataFrame(comparison_data))
    
    # Plot comparison of SMA and EMA for Asset1
    st.markdown("#### Performance Comparison: SMA vs EMA for Asset1")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(sma_results['Asset1']['portfolio_df']['day'], 
            sma_results['Asset1']['portfolio_df']['portfolio_value'], 
            label='SMA Strategy')
    
    ax.plot(ema_results['Asset1']['portfolio_df']['day'], 
            ema_results['Asset1']['portfolio_df']['portfolio_value'], 
            label='EMA Strategy')
    
    ax.set_title('Asset1: SMA vs EMA Strategy Performance')
    ax.set_xlabel('Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Parameter optimization comparison
    st.subheader("Parameter Optimization Comparison")
    
    # Define parameter sets
    parameter_sets = [
        (12, 26, 0.005),  # Set (a)
        (5, 10, 0.005),   # Set (b)
        (10, 20, 0.005)   # Set (c)
    ]
    
    # Run parameter comparison for both strategies
    sma_param_results = run_strategy_parameter_comparison(data, 'Asset1', parameter_sets, strategy_type="SMA")
    ema_param_results = run_strategy_parameter_comparison(data, 'Asset1', parameter_sets, strategy_type="EMA")
    
    # Create comparison table
    param_comparison_data = []
    for i, params in enumerate(parameter_sets):
        short_window, long_window, delta = params
        param_key = f"Set {chr(97 + i)}: S={short_window}, L={long_window}, δ={delta}"
        
        param_comparison_data.append({
            'Parameter Set': param_key,
            'SMA Final Value': f"${sma_param_results[list(sma_param_results.keys())[i]]['final_value']:.2f}",
            'SMA Return (%)': f"{sma_param_results[list(sma_param_results.keys())[i]]['return_pct']:.2f}%",
            'SMA Trades': len(sma_param_results[list(sma_param_results.keys())[i]]['trades_df']),
            'EMA Final Value': f"${ema_param_results[list(ema_param_results.keys())[i]]['final_value']:.2f}",
            'EMA Return (%)': f"{ema_param_results[list(ema_param_results.keys())[i]]['return_pct']:.2f}%",
            'EMA Trades': len(ema_param_results[list(ema_param_results.keys())[i]]['trades_df'])
        })
    
    st.table(pd.DataFrame(param_comparison_data))
    
    # Conclusion
    st.subheader("Conclusion")
    
    st.markdown("""
    In this analysis, we implemented and evaluated two types of MACD trading strategies: one using Simple Moving Averages (SMA) and another using Exponential Moving Averages (EMA). We tested these strategies on four different assets and compared various parameter sets to identify the most profitable configurations.
    
    ### Key Findings
    
    1. **Asset Performance**:
       - Asset1 consistently outperformed the other assets in both SMA and EMA strategies.
       - The ranking of assets varied slightly between the two strategies, with Asset4 performing better under the EMA strategy.
    
    2. **Strategy Comparison**:
       - For Asset1, the SMA strategy with parameters S=10, L=20, δ=0.005 (set c) yielded the highest overall return of 74.46%.
       - The EMA strategy generally resulted in fewer trades compared to the SMA strategy, which could be advantageous in terms of transaction costs.
    
    3. **Parameter Optimization**:
       - Parameter optimization significantly improved the performance of both strategies.
       - For the SMA strategy, set (c) with S=10, L=20 was optimal for Asset1.
       - For the EMA strategy, set (b) with S=5, L=10 was optimal for Asset1.
    
    ### Practical Implications
    
    The results of this analysis have several practical implications for algorithmic trading:
    
    1. **Strategy Selection**: The choice between SMA and EMA strategies should be based on the specific characteristics of the asset being traded. For Asset1, the SMA strategy was more effective, while for Asset4, the EMA strategy showed better results.
    
    2. **Parameter Optimization**: Optimizing the parameters of a trading strategy can significantly improve its performance. Different assets may require different parameter sets for optimal results.
    
    3. **Trade Frequency**: The EMA strategy generally resulted in fewer trades, which could be advantageous in terms of transaction costs. However, the SMA strategy with optimized parameters yielded higher returns for Asset1.
    
    4. **Asset Selection**: Asset1 consistently outperformed the other assets, highlighting the importance of asset selection in algorithmic trading.
    
    In conclusion, both SMA and EMA strategies can be effective for algorithmic trading, but their performance depends on the specific asset being traded and the parameters used. Careful optimization of these parameters is essential for maximizing trading profits.
    """)

# Generate PDF button
st.sidebar.markdown("---")
if st.sidebar.button("Generate PDF Report"):
    st.sidebar.info("This would generate a PDF report of the analysis in a real deployment on Streamlit app.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by: Dymasius Yusuf Sitepu (G2303593E - dymasius001@e.ntu.edu.sg / e0196756@u.nus.edu) - NTU Algotrading & Robo Advisor")
st.sidebar.markdown("© 2025 All Rights Reserved")
