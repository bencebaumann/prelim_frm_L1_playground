#imports
import pandas as pd
import os
import sys
import datetime
from dateutil.relativedelta import relativedelta
from efficient_frontier import EfficientFrontier

# Name of the target directory (case-sensitive)
TARGET_DIR = "FRM" #project root

# Start from the current script's directory
FRM_dir = os.path.abspath(os.path.dirname(__file__))

# Roll up until we hit the target directory or the root
while True:
    if os.path.basename(FRM_dir) == TARGET_DIR:
        break
    parent = os.path.dirname(FRM_dir)
    if parent == FRM_dir:
        raise FileNotFoundError(f"Directory '{TARGET_DIR}' not found in path hierarchy.")
    FRM_dir = parent

# Add FRM to sys.path in order to enable import from its submodules via my_utils.simple_historical
if FRM_dir not in sys.path:
    sys.path.append(FRM_dir)

from my_utils.simple_historical import HistoricalEstimator


if __name__ == '__main__':
    tickers = [
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "MMM",    # 3M Company
    "GOOGL",  # Alphabet Inc. (Google)
    "AMZN",   # Amazon.com Inc.
    "META",   # Meta Platforms Inc.
    "TSLA",   # Tesla Inc.
    "BRK-B",  # Berkshire Hathaway Inc. (Class B)
    "AVGO",   # Broadcom Inc.
    "WMT"     # Walmart Inc.
    ]  # List of tickers for the basket considered for the portfolio


    rebalancing_date = datetime.date(2025, 3, 15) #when the portfolio rebalance occurs
    lookback = 10            # years for the same months ahead
    horizon = 3             # months to look ahead
    risk_free_rate = 0.03   # 3% risk-free rate, used for calculating sharpe ratio as per:
    """
    sharpe ratio = ( portfolio return - risk free return ) / risk measure <- measures how effectively an allocation
    converts risk to positive returns
    """
    risk_limit = None       # for EfficientFrontier upper bound



    estimate = HistoricalEstimator(tickers=tickers, rebalance_date=rebalancing_date, lookback=lookback, horizon=horizon)


    
    estimate.estimate_returns()
    returns = estimate.returns
    returns = {k: v.iloc[0].item() if isinstance(v, pd.Series) else float(v) for k, v in returns.items()} 
    print(returns)

    estimate.estimate_risks()
    risk = estimate.risks
    print(risk)
    estimate.estimate_covariance()
    covariance_matrix = estimate.cov_matrix
    print(covariance_matrix)



    ef = EfficientFrontier(
        estimated_returns=returns,
        covariance_matrix=covariance_matrix,
        estimated_risks=risk,
        risk_free_rate=risk_free_rate,    
        risk_limit=None         
    )
    

    
    # Find optimal portfolios
    print("Finding optimal portfolios...")
    min_var = ef.find_minimum_variance_portfolio()
    max_sharpe = ef.find_maximum_sharpe_portfolio()
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    efficient_portfolios = ef.generate_efficient_frontier(num_portfolios=800) #num_portfolios<- how many allocations to calculate
    
    # Display results
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nMaximum Sharpe Portfolio:")
    max_sharpe_allocation = ef.get_portfolio_allocation('max_sharpe')
    for _, row in max_sharpe_allocation.iterrows():
        print(f"  {row['Ticker']}: {row['Weight_Percentage']:.1f}%")
    
    print("\nMinimum Variance(Risk) Portfolio:")
    min_var_allocation = ef.get_portfolio_allocation('min_variance')
    for _, row in min_var_allocation.iterrows():
        print(f"  {row['Ticker']}: {row['Weight_Percentage']:.1f}%")
        

    # Summary statistics
    summary = ef.get_summary_statistics()
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Generate and plot frontier
    # assumeing project is being ran at root -> ...\FRM\...
    plot_save_loc = r'FRM_L1/FRM_L1_FOUNDATIONS_OF_RISK_MANAGEMENT/C5_MODERN_PF_THEORY_AND_CAPM/efficient_frontier_with_rf.png'
    ef.plot_efficient_frontier(figsize=(14, 8), save_path=plot_save_loc)

    #w/o r_free rate
    plot_save_loc = r'FRM_L1/FRM_L1_FOUNDATIONS_OF_RISK_MANAGEMENT/C5_MODERN_PF_THEORY_AND_CAPM/efficient_frontier.png'
    ef.plot_efficient_frontier_no_rf(figsize=(14, 8), save_path=plot_save_loc)