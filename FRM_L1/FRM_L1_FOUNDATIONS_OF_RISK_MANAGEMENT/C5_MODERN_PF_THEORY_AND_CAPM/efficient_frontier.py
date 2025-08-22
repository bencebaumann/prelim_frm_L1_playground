"""
Efficient Frontier Class for Modern Portfolio Theory

This module implements a comprehensive EfficientFrontier class that calculates 
and visualizes the efficient frontier based on Markowitz mean-variance optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Union, Optional, Tuple
import warnings
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

class EfficientFrontier:
    """
    A class to calculate and visualize the efficient frontier based on Modern Portfolio Theory.
    
    This class implements Markowitz's mean-variance optimization framework to:
    - Calculate optimal portfolio weights for different risk-return combinations
    - Generate and plot the efficient frontier
    - Find key portfolios (minimum variance, maximum Sharpe ratio)
    - Validate input data consistency
    - Handle constraints and bounds appropriately
    """
    
    def __init__(self, 
                 estimated_returns: Union[Dict[str, float], List[float]], 
                 covariance_matrix: Union[pd.DataFrame, np.ndarray],
                 estimated_risks: Union[Dict[str, float], List[float]] = None,
                 risk_free_rate: float = 0.02,
                 risk_limit: Optional[float] = None):
        """
        Initialize the EfficientFrontier class.
        
        Parameters:
        -----------
        estimated_returns : Dict[str, float] or List[float]
            Expected returns for each asset (annualized)
        covariance_matrix : pd.DataFrame or np.ndarray
            Covariance matrix of asset returns (annualized)
        estimated_risks : Dict[str, float] or List[float], optional
            Expected risks (standard deviations) for each asset
            If None, will be calculated from covariance matrix diagonal
        risk_free_rate : float, default 0.02
            Risk-free rate for Sharpe ratio calculation (annualized)
        risk_limit : float, optional
            Maximum allowed portfolio risk (standard deviation)
        """
        
        self.risk_free_rate = risk_free_rate
        self.risk_limit = risk_limit
        
        # Process and validate inputs
        self._process_inputs(estimated_returns, covariance_matrix, estimated_risks)
        
        # Initialize results storage
        self.efficient_portfolios = []
        self.min_var_portfolio = None
        self.max_sharpe_portfolio = None
        self.tangency_portfolio = None
        
        print("✓ EfficientFrontier initialized successfully")
        print(f"  Assets: {self.n_assets}")
        print(f"  Tickers: {self.tickers}")
        print(f"  Risk-free rate: {self.risk_free_rate:.2%}")
        if self.risk_limit:
            print(f"  Risk limit: {self.risk_limit:.2%}")
    
    def _process_inputs(self, estimated_returns, covariance_matrix, estimated_risks):
        """Process and validate all inputs."""
        
        # Process estimated returns
        if isinstance(estimated_returns, dict):
            self.tickers = list(estimated_returns.keys())
            self.expected_returns = np.array(list(estimated_returns.values()))
        else:
            self.expected_returns = np.array(estimated_returns)
            self.tickers = [f"Asset_{i+1}" for i in range(len(estimated_returns))]
        
        self.n_assets = len(self.expected_returns)
        
        # Process covariance matrix
        if isinstance(covariance_matrix, pd.DataFrame):
            self.cov_matrix = covariance_matrix.values
            if covariance_matrix.index.tolist() != self.tickers:
                warnings.warn("Covariance matrix index doesn't match tickers. Using provided order.")
        else:
            self.cov_matrix = np.array(covariance_matrix)
        
        # Process estimated risks
        if estimated_risks is None:
            # Calculate from covariance matrix diagonal
            self.estimated_risks = np.sqrt(np.diag(self.cov_matrix))
        else:
            if isinstance(estimated_risks, dict):
                self.estimated_risks = np.array([estimated_risks[ticker] for ticker in self.tickers])
            else:
                self.estimated_risks = np.array(estimated_risks)
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Perform comprehensive input validation."""
        
        # Check dimensions
        if len(self.expected_returns) != self.n_assets:
            raise ValueError(f"Expected returns length ({len(self.expected_returns)}) doesn't match number of assets ({self.n_assets})")
        
        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Covariance matrix shape {self.cov_matrix.shape} doesn't match number of assets ({self.n_assets})")
        
        if len(self.estimated_risks) != self.n_assets:
            raise ValueError(f"Estimated risks length ({len(self.estimated_risks)}) doesn't match number of assets ({self.n_assets})")
        
        # Check covariance matrix properties
        if not np.allclose(self.cov_matrix, self.cov_matrix.T):
            warnings.warn("Covariance matrix is not symmetric. Making it symmetric.")
            self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(self.cov_matrix)
        if np.any(eigenvalues < -1e-8):
            warnings.warn("Covariance matrix is not positive semi-definite. Adding regularization.")
            self.cov_matrix += np.eye(self.n_assets) * 1e-6
        
        # Validate risk consistency
        cov_risks = np.sqrt(np.diag(self.cov_matrix))
        risk_diff = np.abs(self.estimated_risks - cov_risks)
        if np.any(risk_diff > 0.01):  # 1% tolerance
            warnings.warn("Estimated risks don't match covariance matrix diagonal. Using covariance matrix values.")
            self.estimated_risks = cov_risks
        
        # Check for reasonable values
        if np.any(self.expected_returns < -1) or np.any(self.expected_returns > 5):
            warnings.warn("Some expected returns seem unrealistic (< -100% or > 500%)")
        
        if np.any(self.estimated_risks < 0) or np.any(self.estimated_risks > 2):
            warnings.warn("Some risk values seem unrealistic (< 0% or > 200%)")
        
        print("✓ Input validation completed")
    
    def _portfolio_performance(self, weights):
        """Calculate portfolio return and risk."""
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_risk
    
    def _negative_sharpe_ratio(self, weights):
        """Objective function to minimize (negative Sharpe ratio)."""
        portfolio_return, portfolio_risk = self._portfolio_performance(weights)
        if portfolio_risk == 0:
            return -np.inf
        return -(portfolio_return - self.risk_free_rate) / portfolio_risk
    
    def _portfolio_variance(self, weights):
        """Objective function for minimum variance optimization."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def find_minimum_variance_portfolio(self):
        """Find the portfolio with minimum variance."""
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Bounds: no short selling (weights >= 0), no single asset > 100%
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self._portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            portfolio_return, portfolio_risk = self._portfolio_performance(weights)
            
            self.min_var_portfolio = {
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk
            }
            
            print(f"✓ Minimum variance portfolio found:")
            print(f"  Expected return: {portfolio_return:.2%}")
            print(f"  Risk (std dev): {portfolio_risk:.2%}")
            print(f"  Sharpe ratio: {self.min_var_portfolio['sharpe_ratio']:.3f}")
        else:
            raise RuntimeError("Failed to find minimum variance portfolio")
        
        return self.min_var_portfolio
    
    def find_maximum_sharpe_portfolio(self):
        """Find the portfolio with maximum Sharpe ratio (tangency portfolio)."""
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self._negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            portfolio_return, portfolio_risk = self._portfolio_performance(weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            self.max_sharpe_portfolio = {
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio
            }
            
            # This is also the tangency portfolio
            self.tangency_portfolio = self.max_sharpe_portfolio.copy()
            
            print(f"✓ Maximum Sharpe ratio portfolio found:")
            print(f"  Expected return: {portfolio_return:.2%}")
            print(f"  Risk (std dev): {portfolio_risk:.2%}")
            print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
        else:
            raise RuntimeError("Failed to find maximum Sharpe ratio portfolio")
        
        return self.max_sharpe_portfolio
    
    def generate_efficient_frontier(self, num_portfolios=1000):
        """Generate the efficient frontier by optimizing for different target returns."""
        
        # First find key portfolios
        if self.min_var_portfolio is None:
            self.find_minimum_variance_portfolio()
        if self.max_sharpe_portfolio is None:
            self.find_maximum_sharpe_portfolio()
        
        # Define return range from min variance portfolio to highest single asset return
        min_ret = self.min_var_portfolio['return']
        max_ret = max(self.expected_returns) * 0.95  # Slightly less than max for feasibility
        
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Constraints: weights sum to 1, target return achieved
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                {'type': 'eq', 'fun': lambda weights, target=target_return: 
                 np.sum(weights * self.expected_returns) - target}
            ]
            
            # Additional constraint for risk limit if specified
            if self.risk_limit:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda weights: self.risk_limit**2 - np.dot(weights.T, np.dot(self.cov_matrix, weights))
                })
            
            # Bounds: no short selling
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            
            # Initial guess: equal weights
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            # Optimize
            result = minimize(
                self._portfolio_variance,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return, portfolio_risk = self._portfolio_performance(weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                
                efficient_portfolios.append({
                    'weights': weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'target_return': target_return
                })
        
        self.efficient_portfolios = efficient_portfolios
        print(f"✓ Generated efficient frontier with {len(efficient_portfolios)} portfolios")
        
        return efficient_portfolios
    
    def get_portfolio_allocation(self, portfolio_type='max_sharpe'):
        """
        Get detailed portfolio allocation information.
        
        Parameters:
        -----------
        portfolio_type : str, default 'max_sharpe'
            Type of portfolio: 'max_sharpe', 'min_variance', or portfolio index
        
        Returns:
        --------
        pd.DataFrame
            Portfolio allocation details
        """
        
        if portfolio_type == 'max_sharpe':
            if self.max_sharpe_portfolio is None:
                self.find_maximum_sharpe_portfolio()
            portfolio = self.max_sharpe_portfolio
        elif portfolio_type == 'min_variance':
            if self.min_var_portfolio is None:
                self.find_minimum_variance_portfolio()
            portfolio = self.min_var_portfolio
        else:
            # Assume it's an index for efficient portfolios
            if not self.efficient_portfolios:
                self.generate_efficient_frontier()
            portfolio = self.efficient_portfolios[portfolio_type]
        
        allocation_df = pd.DataFrame({
            'Ticker': self.tickers,
            'Weight': portfolio['weights'],
            'Expected_Return': self.expected_returns,
            'Risk': self.estimated_risks,
            'Contribution_to_Return': portfolio['weights'] * self.expected_returns,
            'Weight_Percentage': portfolio['weights'] * 100
        })
        
        allocation_df = allocation_df.round(4)
        allocation_df = allocation_df.sort_values('Weight', ascending=False)
        
        return allocation_df
    
    def get_summary_statistics(self):
        """Get summary statistics for the efficient frontier analysis."""
        
        if not self.efficient_portfolios:
            self.generate_efficient_frontier()
        
        risks = [p['risk'] for p in self.efficient_portfolios]
        returns = [p['return'] for p in self.efficient_portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in self.efficient_portfolios]
        
        summary = {
            'Number of Assets': self.n_assets,
            'Risk-Free Rate': self.risk_free_rate,
            'Efficient Portfolios Generated': len(self.efficient_portfolios),
            'Risk Range': f"{min(risks):.2%} - {max(risks):.2%}",
            'Return Range': f"{min(returns):.2%} - {max(returns):.2%}",
            'Sharpe Ratio Range': f"{min(sharpe_ratios):.3f} - {max(sharpe_ratios):.3f}",
            'Best Sharpe Ratio': max(sharpe_ratios) if sharpe_ratios else 'Not calculated',
            'Min Variance Portfolio Risk': self.min_var_portfolio['risk'] if self.min_var_portfolio else 'Not calculated',
            'Max Sharpe Portfolio Risk': self.max_sharpe_portfolio['risk'] if self.max_sharpe_portfolio else 'Not calculated'
        }
        
        return summary
    
    def plot_efficient_frontier(self, figsize=(12, 8), save_path=None):
        """
        Plot the efficient frontier with key portfolios and capital allocation line.
        
        Parameters:
        -----------
        figsize : tuple, default (12, 8)
            Figure size for the plot
        save_path : str, optional
            Path to save the plot
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object for further customization if needed
        """
        
        if not self.efficient_portfolios:
            print("Generating efficient frontier first...")
            self.generate_efficient_frontier()
        
        # Extract data for plotting
        risks = [p['risk'] for p in self.efficient_portfolios]
        returns = [p['return'] for p in self.efficient_portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in self.efficient_portfolios]
        
        # Create the plot
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # Main plot
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Plot efficient frontier with color gradient based on Sharpe ratio
        scatter = ax_main.scatter(risks, returns, c=sharpe_ratios, cmap='viridis', 
                                s=50, alpha=0.7, label='Efficient Frontier')
        
        # Plot individual assets
        ax_main.scatter(self.estimated_risks, self.expected_returns, 
                    c='red', s=100, alpha=0.8, marker='D', 
                    label='Individual Assets')
        
        # Annotate individual assets
        for i, ticker in enumerate(self.tickers):
            ax_main.annotate(ticker, 
                        (self.estimated_risks[i], self.expected_returns[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Plot key portfolios
        if self.min_var_portfolio:
            ax_main.scatter(self.min_var_portfolio['risk'], self.min_var_portfolio['return'],
                        c='blue', s=150, marker='*', 
                        label=f"Min Variance (Sharpe: {self.min_var_portfolio['sharpe_ratio']:.2f})")
        
        if self.max_sharpe_portfolio:
            ax_main.scatter(self.max_sharpe_portfolio['risk'], self.max_sharpe_portfolio['return'],
                        c='gold', s=150, marker='*', 
                        label=f"Max Sharpe (Sharpe: {self.max_sharpe_portfolio['sharpe_ratio']:.2f})")
        
        # Plot Capital Allocation Line (CAL) if tangency portfolio exists
        if self.tangency_portfolio:
            # CAL extends from risk-free rate to tangency portfolio and beyond
            tangency_risk = self.tangency_portfolio['risk']
            tangency_return = self.tangency_portfolio['return']
            
            # Extend the line beyond the tangency portfolio
            max_risk_for_cal = max(risks) * 1.2
            cal_risks = np.linspace(0, max_risk_for_cal, 100)
            cal_returns = self.risk_free_rate + (tangency_return - self.risk_free_rate) * cal_risks / tangency_risk
            
            ax_main.plot(cal_risks, cal_returns, 'r--', linewidth=2, alpha=0.7,
                        label=f'Capital Allocation Line (RF: {self.risk_free_rate:.1%})')
        
        # Add risk limit line if specified
        if self.risk_limit:
            ax_main.axvline(x=self.risk_limit, color='orange', linestyle=':', 
                        linewidth=2, label=f'Risk Limit: {self.risk_limit:.1%}')
        
        ax_main.set_xlabel('Risk (Standard Deviation)', fontsize=12)
        ax_main.set_ylabel('Expected Return', fontsize=12)
        ax_main.set_title('Efficient Frontier - Modern Portfolio Theory', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format axes as percentages
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Risk histogram (bottom subplot)
        ax_risk = fig.add_subplot(gs[1, 0])
        ax_risk.hist(risks, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax_risk.set_xlabel('Risk (Standard Deviation)', fontsize=10)
        ax_risk.set_ylabel('Frequency', fontsize=10)
        ax_risk.set_title('Risk Distribution', fontsize=12)
        ax_risk.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        
        # Return histogram (right subplot)
        ax_return = fig.add_subplot(gs[0, 1])
        ax_return.hist(returns, bins=20, alpha=0.7, color='lightcoral', 
                    edgecolor='black', orientation='horizontal')
        ax_return.set_ylabel('Expected Return', fontsize=10)
        ax_return.set_xlabel('Frequency', fontsize=10)
        ax_return.set_title('Return Distribution', fontsize=12)
        ax_return.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig
    

    def plot_efficient_frontier_no_rf(self, figsize=(12, 8), save_path=None):
        """
        Plot the efficient frontier (Markowitz bullet) without risk-free rate or CAL.

        Parameters:
        -----------
        figsize : tuple, default (12, 8)
            Figure size for the plot
        save_path : str, optional
            Path to save the plot

        Returns:
        --------
        fig : matplotlib.                                                   figure.Figure
            The figure object for further customization if needed
        """
        if not self.efficient_portfolios:
            print("Generating efficient frontier first...")
            self.generate_efficient_frontier()

        # Extract data for plotting
        risks = [p['risk'] for p in self.efficient_portfolios]
        returns = [p['return'] for p in self.efficient_portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in self.efficient_portfolios]

        # Create the plot
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

        # Main plot
        ax_main = fig.add_subplot(gs[0, 0])

        # Plot efficient frontier with color gradient based on Sharpe ratio
        scatter = ax_main.scatter(risks, returns, c=sharpe_ratios, cmap='viridis',
                                s=50, alpha=0.7, label='Efficient Frontier')

        # Plot individual assets
        ax_main.scatter(self.estimated_risks, self.expected_returns,
                        c='red', s=100, alpha=0.8, marker='D',
                        label='Individual Assets')

        # Annotate individual assets
        for i, ticker in enumerate(self.tickers):
            ax_main.annotate(ticker,
                            (self.estimated_risks[i], self.expected_returns[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')

        # Plot key portfolios
        if self.min_var_portfolio:
            ax_main.scatter(self.min_var_portfolio['risk'], self.min_var_portfolio['return'],
                            c='blue', s=150, marker='*',
                            label=f"Min Variance (Sharpe: {self.min_var_portfolio['sharpe_ratio']:.2f})")

        if self.max_sharpe_portfolio:
            ax_main.scatter(self.max_sharpe_portfolio['risk'], self.max_sharpe_portfolio['return'],
                            c='gold', s=150, marker='*',
                            label=f"Max Sharpe (Sharpe: {self.max_sharpe_portfolio['sharpe_ratio']:.2f})")

        # Add risk limit line if specified
        if self.risk_limit:
            ax_main.axvline(x=self.risk_limit, color='orange', linestyle=':',
                            linewidth=2, label=f'Risk Limit: {self.risk_limit:.1%}')

        ax_main.set_xlabel('Risk (Standard Deviation)', fontsize=12)
        ax_main.set_ylabel('Expected Return', fontsize=12)
        ax_main.set_title('Efficient Frontier (No Risk-Free Rate)', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format axes as percentages
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

        # Colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('Sharpe Ratio', fontsize=12)

        # Risk histogram (bottom subplot)
        ax_risk = fig.add_subplot(gs[1, 0])
        ax_risk.hist(risks, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax_risk.set_xlabel('Risk (Standard Deviation)', fontsize=10)
        ax_risk.set_ylabel('Frequency', fontsize=10)
        ax_risk.set_title('Risk Distribution', fontsize=12)
        ax_risk.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

        # Return histogram (right subplot)
        ax_return = fig.add_subplot(gs[0, 1])
        ax_return.hist(returns, bins=20, alpha=0.7, color='lightcoral',
                    edgecolor='black', orientation='horizontal')
        ax_return.set_ylabel('Expected Return', fontsize=10)
        ax_return.set_xlabel('Frequency', fontsize=10)
        ax_return.set_title('Return Distribution', fontsize=12)
        ax_return.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()
        return fig

   








if __name__ == "__main__":
    # Example usage demonstration
    print("=" * 60)
    print("EFFICIENT FRONTIER CLASS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    
    # Updated sample data with BTC and MMM
    estimated_returns = {
        'AAPL': 0.12,    # Tech - Moderate growth
        'MSFT': 0.10,    # Tech - Stable giant
        'GOOGL': 0.15,   # Tech - Growth potential  
        'TSLA': 0.20,    # EV - High volatility
        'BTC': 0.25,     # Crypto - High risk/reward
        'MMM': 0.05      # Industrial - Stable dividend
    }

    # Annualized standard deviations (risks)
    risks = np.array([
        0.20,  # AAPL
        0.18,  # MSFT
        0.25,  # GOOGL
        0.35,  # TSLA
        0.75,  # BTC (Bitcoin's typical volatility)
        0.22   # MMM (3M's historical volatility)
    ])

    # Realistic correlation matrix (6x6)
    correlation_matrix = np.array([
        # AAPL   MSFT   GOOGL  TSLA   BTC    MMM
        [1.00,   0.60,  0.70,  0.40,  0.20,  0.50],  # AAPL
        [0.60,   1.00,  0.65,  0.35,  0.20,  0.50],  # MSFT
        [0.70,   0.65,  1.00,  0.45,  0.20,  0.50],  # GOOGL
        [0.40,   0.35,  0.45,  1.00,  0.20,  0.50],  # TSLA
        [0.20,   0.20,  0.20,  0.20,  1.00,  0.10],  # BTC (low stock correlation)
        [0.50,   0.50,  0.50,  0.50,  0.10,  1.00]   # MMM (industrial correlation)
    ])

    
    # Convert to covariance matrix
    D = np.diag(risks)
    covariance_matrix = D @ correlation_matrix @ D
    
    print("Sample Portfolio Data:")
    print("Assets: " + ", ".join(estimated_returns))
    print("Expected Returns:", [f"{r:.1%}" for r in estimated_returns.values()])
    print("Expected Risks:", [f"{r:.1%}" for r in risks])
    print()
    
    # Initialize EfficientFrontier
    ef = EfficientFrontier(
        estimated_returns=estimated_returns,
        covariance_matrix=covariance_matrix,
        estimated_risks=risks,
        risk_free_rate=0.03,    # 3% risk-free rate
        risk_limit=None         # 25% maximum risk
    )
    
    print()
    
    # Find optimal portfolios
    print("Finding optimal portfolios...")
    min_var = ef.find_minimum_variance_portfolio()
    max_sharpe = ef.find_maximum_sharpe_portfolio()
    
    print()
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    efficient_portfolios = ef.generate_efficient_frontier(num_portfolios=50)
    
    print()
    
    # Display results
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nMaximum Sharpe Portfolio:")
    max_sharpe_allocation = ef.get_portfolio_allocation('max_sharpe')
    for _, row in max_sharpe_allocation.iterrows():
        print(f"  {row['Ticker']}: {row['Weight_Percentage']:.1f}%")
    
    print("\nMinimum Variance Portfolio:")
    min_var_allocation = ef.get_portfolio_allocation('min_variance')
    for _, row in min_var_allocation.iterrows():
        print(f"  {row['Ticker']}: {row['Weight_Percentage']:.1f}%")
    
    print()
    
    # Summary statistics
    summary = ef.get_summary_statistics()
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    ef.find_minimum_variance_portfolio()
    ef.find_maximum_sharpe_portfolio()
    
    #assumeing script is being ran at project root <- ...\FRM\...
    plot_save_loc = r'FRM_L1/FRM_L1_FOUNDATIONS_OF_RISK_MANAGEMENT/C5_MODERN_PF_THEORY_AND_CAPM/efficient_frontier_sample.png'

    # Generate and plot frontier
    ef.generate_efficient_frontier(num_portfolios=1000)
    ef.plot_efficient_frontier(figsize=(14, 8), save_path=plot_save_loc)
