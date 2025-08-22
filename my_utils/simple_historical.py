import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from yfinance import download
import numpy as np
from typing import Dict, List, Tuple, Optional

class HistoricalEstimator:
    def __init__(self, tickers: List[str], rebalance_date: datetime.date, 
                 horizon: int, lookback: int, annualize: bool = False):
        self.tickers = tickers
        self.rebalance_date = rebalance_date
        self.horizon = horizon
        self.lookback = lookback
        self.annualize = annualize
        
        # Results storage
        self.returns = None
        self.risks = None
        self.cov_matrix = None
        self.available_tickers = None
        self.price_data = None
        
    def _download_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download all price data once for all periods and tickers.
        Returns a dictionary with successful downloads only.
        """
        print("Downloading historical price data...")
        successful_data = {}
        failed_tickers = []
        
        # Calculate all required date ranges
        date_ranges = []
        for n in range(1, self.lookback + 1):
            start = self.rebalance_date - relativedelta(years=n)
            end = start + relativedelta(months=self.horizon)
            date_ranges.append((start, end, n))
        
        # Download data for each ticker IN ORIGINAL ORDER
        for ticker in self.tickers:
            ticker_data = {}
            success_count = 0
            
            for start, end, period in date_ranges:
                try:
                    # Check if dates are not in future
                    if start > datetime.date.today() or end > datetime.date.today():
                        continue
                        
                    prices = download(ticker, start=start, end=end, progress=False)['Close']
                    
                    # Check data sufficiency (minimum 10 data points)
                    if len(prices) >= 10:
                        ticker_data[period] = {
                            'prices': prices,
                            'start': start,
                            'end': end,
                            'return': (prices.iloc[-1].item() / prices.iloc[0].item()) - 1
                        }
                        success_count += 1
                    
                except Exception as e:
                    print(f"Failed to download {ticker} for period {period}: {str(e)}")
                    continue
            
            # Only include ticker if we have at least 2 successful periods
            if success_count >= 2:
                successful_data[ticker] = ticker_data
            else:
                failed_tickers.append(ticker)
                print(f"Insufficient data for {ticker} - excluding from analysis")
        
        if failed_tickers:
            print(f"Excluded tickers due to insufficient data: {failed_tickers}")
        
        if not successful_data:
            raise ValueError("No tickers have sufficient historical data")
            
        print(f"Successfully loaded data for {len(successful_data)} tickers")
        
        return successful_data
    
    def _calculate_returns(self, data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate average historical returns for each ticker."""
        returns = {}
        
        # Process tickers in original order to maintain consistency
        for ticker in self.tickers:
            if ticker in data:
                periods = data[ticker]
                period_returns = [period_data['return'] for period_data in periods.values()]
                
                if len(period_returns) >= 2:
                    avg_return = sum(period_returns) / len(period_returns)
                    returns[ticker] = float(avg_return)
        
        return returns
    
    def _calculate_risks(self, data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate historical volatility (standard deviation) for each ticker."""
        risks = {}
        
        # Process tickers in original order to maintain consistency
        for ticker in self.tickers:
            if ticker in data:
                periods = data[ticker]
                period_returns = [period_data['return'] for period_data in periods.values()]
                
                if len(period_returns) >= 2:
                    volatility = float(pd.Series(period_returns).std())
                    risks[ticker] = volatility
        
        return risks
    
    def _calculate_covariance_matrix(self, data: Dict[str, Dict]) -> pd.DataFrame:
        """Calculate covariance matrix from the downloaded data."""
        # Create returns matrix for all periods and tickers
        returns_matrix = {}
        
        # Get all available periods
        all_periods = set()
        for periods in data.values():
            all_periods.update(periods.keys())
        
        # Build returns matrix ensuring consistent periods across tickers
        # Process in original ticker order for consistency
        for period in sorted(all_periods):
            period_returns = {}
            for ticker in self.tickers:  # CRITICAL: Use original ticker order
                if ticker in data and period in data[ticker]:
                    period_returns[ticker] = data[ticker][period]['return']
            
            # Only include periods where we have data for multiple tickers
            if len(period_returns) >= 2:
                for ticker, ret in period_returns.items():
                    if ticker not in returns_matrix:
                        returns_matrix[ticker] = []
                    returns_matrix[ticker].append(ret)
        
        # Ensure all tickers have same number of observations
        min_observations = min(len(returns) for returns in returns_matrix.values())
        for ticker in returns_matrix:
            returns_matrix[ticker] = returns_matrix[ticker][:min_observations]
        
        # Create DataFrame with columns in original ticker order
        ordered_tickers = [t for t in self.tickers if t in returns_matrix]
        df = pd.DataFrame({ticker: returns_matrix[ticker] for ticker in ordered_tickers})
        
        if df.empty:
            raise ValueError("No overlapping data for covariance calculation")
        
        # Calculate covariance matrix
        cov_matrix = df.cov()
        
        # Apply annualization factor if requested
        if self.annualize:
            ann_factor = 12 / self.horizon
            cov_matrix = cov_matrix * ann_factor
        
        return cov_matrix
    
    def estimate_all(self) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
        """
        Download data once and calculate all three metrics.
        Returns: (returns_dict, risks_dict, covariance_matrix)
        
        FIXED VERSION: Preserves original ticker ordering consistently.
        """
        # Download all data once
        self.price_data = self._download_all_data()
        
        # Calculate all metrics from the single dataset
        self.returns = self._calculate_returns(self.price_data)
        self.risks = self._calculate_risks(self.price_data)
        self.cov_matrix = self._calculate_covariance_matrix(self.price_data)
        
        # CRITICAL FIX: Preserve original ticker order
        returns_tickers = set(self.returns.keys())
        risks_tickers = set(self.risks.keys()) 
        cov_tickers = set(self.cov_matrix.columns)
        common_tickers_set = returns_tickers & risks_tickers & cov_tickers
        
        # Filter self.tickers to preserve original order
        common_tickers_ordered = [t for t in self.tickers if t in common_tickers_set]
        
        # Apply consistent ordering to all data structures
        self.returns = {k: self.returns[k] for k in common_tickers_ordered}
        self.risks = {k: self.risks[k] for k in common_tickers_ordered}
        self.cov_matrix = self.cov_matrix.loc[common_tickers_ordered, common_tickers_ordered]
        
        # Update available tickers to final ordered set
        self.available_tickers = common_tickers_ordered
        
        print(f"Final analysis includes {len(self.available_tickers)} tickers: {self.available_tickers}")
        
        return self.returns, self.risks, self.cov_matrix
    
    # Legacy methods for backward compatibility
    def estimate_returns(self):
        """Legacy method - use estimate_all() instead for better performance."""
        if self.returns is None:
            self.estimate_all()
        return self.returns
    
    def estimate_risks(self):
        """Legacy method - use estimate_all() instead for better performance."""
        if self.risks is None:
            self.estimate_all()
        return self.risks
    
    def estimate_covariance(self):
        """Legacy method - use estimate_all() instead for better performance."""
        if self.cov_matrix is None:
            self.estimate_all()
        return self.cov_matrix
    
    def validate_ticker_consistency(self) -> bool:
        """
        Validate that all data structures have consistent ticker ordering.
        """
        if not all([self.returns, self.risks, self.cov_matrix is not None]):
            raise ValueError("Must run estimate_all() first")
            
        returns_tickers = list(self.returns.keys())
        risks_tickers = list(self.risks.keys())
        cov_tickers = list(self.cov_matrix.columns)
        
        is_consistent = (returns_tickers == risks_tickers == cov_tickers)
        
        if is_consistent:
            print("✓ Ticker ordering is consistent across all data structures")
        else:
            print("✗ Ticker ordering inconsistency detected!")
            print(f"  Returns: {returns_tickers}")
            print(f"  Risks: {risks_tickers}")
            print(f"  Covariance: {cov_tickers}")
            
        return is_consistent


# Standalone functions for backward compatibility
def historic_monthly_return_estimate(tickers, rebalance_date, horizon: int, lookback: int):
    """Legacy function - use HistoricalEstimator.estimate_all() for better performance."""
    estimator = HistoricalEstimator(tickers, rebalance_date, horizon, lookback)
    returns, _, _ = estimator.estimate_all()
    return returns

def historic_monthly_risk_estimate(tickers, rebalance_date, horizon: int, lookback: int):
    """Legacy function - use HistoricalEstimator.estimate_all() for better performance."""
    estimator = HistoricalEstimator(tickers, rebalance_date, horizon, lookback)
    _, risks, _ = estimator.estimate_all()
    return risks

def calculate_covariance_matrix(tickers, rebalance_date, horizon: int, lookback: int, annualize=False):
    """Legacy function - use HistoricalEstimator.estimate_all() for better performance."""
    estimator = HistoricalEstimator(tickers, rebalance_date, horizon, lookback, annualize)
    _, _, cov_matrix = estimator.estimate_all()
    return cov_matrix


# Testing function to verify consistency
def test_ticker_consistency(tickers, rebalance_date, horizon, lookback, num_runs=3):
    """
    Test that ticker ordering is consistent across multiple runs.
    """
    print(f"Testing ticker consistency across {num_runs} runs...")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}:")
        estimator = HistoricalEstimator(tickers, rebalance_date, horizon, lookback)
        returns, risks, cov_matrix = estimator.estimate_all()
        
        result = {
            'returns_tickers': list(returns.keys()),
            'risks_tickers': list(risks.keys()),
            'cov_tickers': list(cov_matrix.columns),
            'available_tickers': estimator.available_tickers
        }
        all_results.append(result)
    
    # Check consistency across runs
    first_run = all_results[0]
    is_consistent = True
    
    for i, run_result in enumerate(all_results[1:], 2):
        for key in first_run.keys():
            if first_run[key] != run_result[key]:
                print(f"✗ Inconsistency detected in run {i} for {key}")
                print(f"  Run 1: {first_run[key]}")
                print(f"  Run {i}: {run_result[key]}")
                is_consistent = False
    
    if is_consistent:
        print("✓ All runs produced consistent ticker ordering!")
        print(f"Consistent ticker order: {first_run['available_tickers']}")
    
    return is_consistent


# Demo usage
if __name__ == '__main__':
    tickers = ['AAPL', 'MMM', 'TSLA', 'MSFT', 'F']
    rebalance_date = datetime.date(2025, 3, 15)
    horizon = 3
    lookback = 3

    # Test the fixed implementation
    print("="*60)
    print("TESTING FIXED TICKER ORDERING IMPLEMENTATION")
    print("="*60)
    
    # Run consistency test
    is_consistent = test_ticker_consistency(tickers, rebalance_date, horizon, lookback, num_runs=3)
    
    if is_consistent:
        print("\n✓ SUCCESS: Ticker ordering problem has been resolved!")
    else:
        print("\n✗ FAILURE: Ticker ordering inconsistency still exists!")