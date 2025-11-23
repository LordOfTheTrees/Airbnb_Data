"""
City Comparison Scatter Plots
Creates scatter plots comparing cities on various metrics including regression coefficients

Usage:
    python create_city_comparison_scatter_plots.py -all
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import io
from scipy.stats import linregress

# Import feature engineering functions
from city_level_analysis import apply_all_feature_engineering
from load_zillow_data import load_all_zillow_data
from explore_census_data import load_census_data, match_city_to_census

warnings.filterwarnings('ignore')

# Note: Encoding handled by individual imported modules

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)


def load_city_data(city_name, base_dir='.', use_detailed=False):
    """Load and prepare data for a single city"""
    city_path = Path(base_dir) / city_name
    
    if not city_path.exists():
        return None
    
    # Determine which file to use
    if use_detailed:
        listings_file = city_path / 'listings.csv.gz'
        if not listings_file.exists():
            listings_file = city_path / 'listings.csv'
    else:
        listings_file = city_path / 'listings.csv'
        if not listings_file.exists():
            listings_file = city_path / 'listings.csv.gz'
    
    if not listings_file.exists():
        return None
    
    # Load data
    if str(listings_file).endswith('.gz'):
        df = pd.read_csv(listings_file, compression='gzip')
    else:
        df = pd.read_csv(listings_file)
    
    # Clean price if present
    if 'price' in df.columns:
        df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    
    # Apply feature engineering
    df = apply_all_feature_engineering(df, city_name, include_zillow=True)
    
    return df


def calculate_regression_coefficients(df, city_name):
    """
    Calculate linear regression coefficients for size variables vs log_price
    
    Returns dict with coefficients for bedrooms, bathrooms, accommodates, beds
    """
    results = {}
    
    # Filter to valid data
    if 'log_price' not in df.columns:
        return None
    
    size_vars = ['bedrooms', 'bathrooms', 'accommodates', 'beds']
    
    for size_var in size_vars:
        if size_var not in df.columns:
            continue
        
        # Filter to valid data and remove outliers
        valid = df[[size_var, 'log_price']].dropna()
        
        # Remove outliers based on size variable
        if size_var == 'bathrooms':
            valid = valid[valid[size_var] <= 8]
        elif size_var == 'bedrooms':
            valid = valid[valid[size_var] <= 10]
        elif size_var == 'beds':
            valid = valid[valid[size_var] <= 20]
        elif size_var == 'accommodates':
            valid = valid[valid[size_var] <= 20]
        
        if len(valid) < 10:  # Need at least 10 data points
            continue
        
        try:
            x = valid[size_var].values
            y = valid['log_price'].values
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            results[f'{size_var}_slope'] = slope
            results[f'{size_var}_r_squared'] = r_value**2
            results[f'{size_var}_p_value'] = p_value
            results[f'{size_var}_n'] = len(valid)
        except:
            continue
    
    return results if results else None


def calculate_city_metrics(df, city_name):
    """Calculate city-level summary metrics"""
    metrics = {
        'city': city_name,
        'total_listings': len(df),
    }
    
    # Pricing (use log_price for transformed data)
    if 'log_price' in df.columns:
        metrics['median_log_price'] = df['log_price'].median()
        metrics['mean_log_price'] = df['log_price'].mean()
    if 'price_clean' in df.columns:
        metrics['median_price'] = df['price_clean'].median()
        metrics['mean_price'] = df['price_clean'].mean()
    
    # Occupancy
    if 'occupancy_rate' in df.columns:
        metrics['median_occupancy'] = df['occupancy_rate'].median()
        metrics['mean_occupancy'] = df['occupancy_rate'].mean()
    
    # Revenue
    if 'est_annual_revenue' in df.columns:
        metrics['median_revenue'] = df['est_annual_revenue'].median()
        metrics['mean_revenue'] = df['est_annual_revenue'].mean()
    
    # ROI
    if 'cash_on_cash_roi' in df.columns:
        metrics['median_roi'] = df['cash_on_cash_roi'].median()
        metrics['mean_roi'] = df['cash_on_cash_roi'].median()
    
    # Professionalization
    if 'host_is_professional' in df.columns:
        metrics['pct_professional'] = (df['host_is_professional'].sum() / len(df)) * 100
        metrics['market_professionalization_score'] = df['market_professionalization_score'].median() if 'market_professionalization_score' in df.columns else np.nan
    
    # Property characteristics
    if 'bedrooms' in df.columns:
        metrics['median_bedrooms'] = df['bedrooms'].median()
    if 'bathrooms' in df.columns:
        metrics['median_bathrooms'] = df['bathrooms'].median()
    if 'accommodates' in df.columns:
        metrics['median_accommodates'] = df['accommodates'].median()
    
    return metrics


def create_city_comparison_plots(city_data, output_dir='city_comparison_outputs'):
    """Create scatter plots comparing cities"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(city_data)
    
    print(f"\n{'='*80}")
    print(f"CREATING CITY COMPARISON SCATTER PLOTS")
    print(f"{'='*80}")
    print(f"Loaded data for {len(df)} cities")
    
    # Create 2x2 figure for regression coefficients comparison (coefficients vs each other)
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Regression Coefficients Comparison\n(How property size affects log price across cities)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    # Helper function to add quadrant labels
    def add_quadrant_labels(ax, plot_data, x_col, y_col, x_label, y_label):
        """Add quadrant labels to scatter plot"""
        x_median = plot_data[x_col].median()
        y_median = plot_data[y_col].median()
        
        # Add quadrant lines
        ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add quadrant labels
        ax.text(0.02, 0.98, f'High {y_label}\nLow {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               verticalalignment='top', ha='left')
        ax.text(0.98, 0.98, f'High {y_label}\nHigh {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
               verticalalignment='top', ha='right')
        ax.text(0.02, 0.02, f'Low {y_label}\nLow {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
               verticalalignment='bottom', ha='left')
        ax.text(0.98, 0.02, f'Low {y_label}\nHigh {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
               verticalalignment='bottom', ha='right')
    
    # Plot 1: Bedrooms coefficient vs Bathrooms coefficient
    ax = axes1[0, 0]
    if 'bedrooms_slope' in df.columns and 'bathrooms_slope' in df.columns:
        plot_data = df[['bedrooms_slope', 'bathrooms_slope']].dropna()
        if len(plot_data) >= 3:
            for idx, row in plot_data.iterrows():
                city = df.loc[idx, 'city']
                ax.scatter(row['bathrooms_slope'], row['bedrooms_slope'], 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
                ax.annotate(city, (row['bathrooms_slope'], row['bedrooms_slope']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.8)
            # Trend line
            x_data = plot_data['bathrooms_slope'].values
            y_data = plot_data['bedrooms_slope'].values
            slope, intercept, r_value, _, _ = linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2)
            add_quadrant_labels(ax, plot_data, 'bathrooms_slope', 'bedrooms_slope', 'Bathrooms Coef', 'Bedrooms Coef')
    ax.set_xlabel('Bathrooms Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Bedrooms Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    if 'bedrooms_slope' in df.columns and 'bathrooms_slope' in df.columns:
        plot_data = df[['bedrooms_slope', 'bathrooms_slope']].dropna()
        if len(plot_data) >= 3:
            x_data = plot_data['bathrooms_slope'].values
            y_data = plot_data['bedrooms_slope'].values
            _, _, r_value, _, _ = linregress(x_data, y_data)
            ax.set_title(f'Bedrooms vs Bathrooms Coefficients\n(r = {r_value:.3f}, R² = {r_value**2:.3f})', fontweight='bold', fontsize=12)
        else:
            ax.set_title('Bedrooms vs Bathrooms Coefficients', fontweight='bold', fontsize=12)
    else:
        ax.set_title('Bedrooms vs Bathrooms Coefficients', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bedrooms coefficient vs Accommodates coefficient
    ax = axes1[0, 1]
    if 'bedrooms_slope' in df.columns and 'accommodates_slope' in df.columns:
        plot_data = df[['bedrooms_slope', 'accommodates_slope']].dropna()
        if len(plot_data) >= 3:
            for idx, row in plot_data.iterrows():
                city = df.loc[idx, 'city']
                ax.scatter(row['accommodates_slope'], row['bedrooms_slope'], 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
                ax.annotate(city, (row['accommodates_slope'], row['bedrooms_slope']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.8)
            # Trend line
            x_data = plot_data['accommodates_slope'].values
            y_data = plot_data['bedrooms_slope'].values
            slope, intercept, r_value, _, _ = linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2)
            add_quadrant_labels(ax, plot_data, 'accommodates_slope', 'bedrooms_slope', 'Accommodates Coef', 'Bedrooms Coef')
    ax.set_xlabel('Accommodates Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Bedrooms Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    if 'bedrooms_slope' in df.columns and 'accommodates_slope' in df.columns:
        plot_data = df[['bedrooms_slope', 'accommodates_slope']].dropna()
        if len(plot_data) >= 3:
            x_data = plot_data['accommodates_slope'].values
            y_data = plot_data['bedrooms_slope'].values
            _, _, r_value, _, _ = linregress(x_data, y_data)
            ax.set_title(f'Bedrooms vs Accommodates Coefficients\n(r = {r_value:.3f}, R² = {r_value**2:.3f})', fontweight='bold', fontsize=12)
        else:
            ax.set_title('Bedrooms vs Accommodates Coefficients', fontweight='bold', fontsize=12)
    else:
        ax.set_title('Bedrooms vs Accommodates Coefficients', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bathrooms coefficient vs Accommodates coefficient
    ax = axes1[1, 0]
    if 'bathrooms_slope' in df.columns and 'accommodates_slope' in df.columns:
        plot_data = df[['bathrooms_slope', 'accommodates_slope']].dropna()
        if len(plot_data) >= 3:
            for idx, row in plot_data.iterrows():
                city = df.loc[idx, 'city']
                ax.scatter(row['accommodates_slope'], row['bathrooms_slope'], 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
                ax.annotate(city, (row['accommodates_slope'], row['bathrooms_slope']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.8)
            # Trend line
            x_data = plot_data['accommodates_slope'].values
            y_data = plot_data['bathrooms_slope'].values
            slope, intercept, r_value, _, _ = linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2)
            add_quadrant_labels(ax, plot_data, 'accommodates_slope', 'bathrooms_slope', 'Accommodates Coef', 'Bathrooms Coef')
    ax.set_xlabel('Accommodates Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Bathrooms Coefficient (vs Log Price)', fontweight='bold', fontsize=11)
    if 'bathrooms_slope' in df.columns and 'accommodates_slope' in df.columns:
        plot_data = df[['bathrooms_slope', 'accommodates_slope']].dropna()
        if len(plot_data) >= 3:
            x_data = plot_data['accommodates_slope'].values
            y_data = plot_data['bathrooms_slope'].values
            _, _, r_value, _, _ = linregress(x_data, y_data)
            ax.set_title(f'Bathrooms vs Accommodates Coefficients\n(r = {r_value:.3f}, R² = {r_value**2:.3f})', fontweight='bold', fontsize=12)
        else:
            ax.set_title('Bathrooms vs Accommodates Coefficients', fontweight='bold', fontsize=12)
    else:
        ax.set_title('Bathrooms vs Accommodates Coefficients', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bedrooms vs Bathrooms (alternative view - can be used for another comparison)
    ax = axes1[1, 1]
    if 'bedrooms_slope' in df.columns and 'bathrooms_slope' in df.columns:
        # Show R² values for each city's regression
        plot_data = df[['bedrooms_r_squared', 'bathrooms_r_squared']].dropna()
        if len(plot_data) >= 3:
            for idx, row in plot_data.iterrows():
                city = df.loc[idx, 'city']
                ax.scatter(row['bathrooms_r_squared'], row['bedrooms_r_squared'], 
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
                ax.annotate(city, (row['bathrooms_r_squared'], row['bedrooms_r_squared']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.8)
            # Trend line
            x_data = plot_data['bathrooms_r_squared'].values
            y_data = plot_data['bedrooms_r_squared'].values
            slope, intercept, r_value, _, _ = linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2)
            # Custom quadrant labels for R² plot
            x_median = plot_data['bathrooms_r_squared'].median()
            y_median = plot_data['bedrooms_r_squared'].median()
            ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(0.02, 0.98, 'Low Bedrooms R²\nHigh Bathrooms R²', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                   verticalalignment='top', ha='left')
            ax.text(0.98, 0.98, 'High Bedrooms R²\nHigh Bathrooms R²', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                   verticalalignment='top', ha='right')
            ax.text(0.02, 0.02, 'Low Bedrooms R²\nLow Bathrooms R²', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                   verticalalignment='bottom', ha='left')
            ax.text(0.98, 0.02, 'High Bedrooms R²\nLow Bathrooms R²', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                   verticalalignment='bottom', ha='right')
    ax.set_xlabel('Bathrooms R² (vs Log Price)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Bedrooms R² (vs Log Price)', fontweight='bold', fontsize=11)
    if 'bedrooms_slope' in df.columns and 'bathrooms_slope' in df.columns:
        plot_data = df[['bedrooms_r_squared', 'bathrooms_r_squared']].dropna()
        if len(plot_data) >= 3:
            x_data = plot_data['bathrooms_r_squared'].values
            y_data = plot_data['bedrooms_r_squared'].values
            _, _, r_value, _, _ = linregress(x_data, y_data)
            ax.set_title(f'Model Fit Comparison: Bedrooms vs Bathrooms\n(r = {r_value:.3f}, R² = {r_value**2:.3f})', fontweight='bold', fontsize=12)
        else:
            ax.set_title('Model Fit Comparison: Bedrooms vs Bathrooms', fontweight='bold', fontsize=12)
    else:
        ax.set_title('Model Fit Comparison: Bedrooms vs Bathrooms', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file1 = output_dir / 'city_regression_coefficients_comparison.png'
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file1}")
    plt.close()
    
    # Create 2x2 figure for city metrics comparisons (using log-transformed data)
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('City-Level Metrics Comparison (Log-Transformed)', fontsize=16, fontweight='bold', y=0.995)
    
    # Helper function to add quadrant labels for metrics
    def add_quadrant_labels_metrics(ax, plot_data, x_col, y_col, x_label, y_label):
        """Add quadrant labels to scatter plot"""
        x_median = plot_data[x_col].median()
        y_median = plot_data[y_col].median()
        
        # Add quadrant lines
        ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add quadrant labels
        ax.text(0.02, 0.98, f'High {y_label}\nLow {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               verticalalignment='top', ha='left')
        ax.text(0.98, 0.98, f'High {y_label}\nHigh {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
               verticalalignment='top', ha='right')
        ax.text(0.02, 0.02, f'Low {y_label}\nLow {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
               verticalalignment='bottom', ha='left')
        ax.text(0.98, 0.02, f'Low {y_label}\nHigh {x_label}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
               verticalalignment='bottom', ha='right')
    
    # Plot 1: Median Log Price vs Total Listings
    ax = axes2[0, 0]
    if 'median_log_price' in df.columns:
        plot_data = df[['median_log_price', 'total_listings']].dropna()
        for idx, row in plot_data.iterrows():
            city = df.loc[idx, 'city']
            ax.scatter(row['total_listings'], row['median_log_price'], 
                      s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.annotate(city, (row['total_listings'], row['median_log_price']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.8)
        add_quadrant_labels_metrics(ax, plot_data, 'total_listings', 'median_log_price', 'Market Size', 'Log Price')
    ax.set_xlabel('Total Listings', fontweight='bold', fontsize=11)
    ax.set_ylabel('Median Log Price', fontweight='bold', fontsize=11)
    ax.set_title('Market Size vs Log Price', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Median ROI vs Median Occupancy
    ax = axes2[0, 1]
    plot_data = df[['median_roi', 'median_occupancy']].dropna().copy()
    plot_data['median_occupancy_pct'] = plot_data['median_occupancy'] * 100
    for idx, row in plot_data.iterrows():
        city = df.loc[idx, 'city']
        ax.scatter(row['median_occupancy_pct'], row['median_roi'], 
                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(city, (row['median_occupancy_pct'], row['median_roi']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold', alpha=0.8)
    add_quadrant_labels_metrics(ax, plot_data, 'median_occupancy_pct', 'median_roi', 'Occupancy', 'ROI')
    ax.set_xlabel('Median Occupancy Rate (%)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Median ROI (%)', fontweight='bold', fontsize=11)
    ax.set_title('Occupancy vs ROI', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Professionalization vs Market Size
    ax = axes2[1, 0]
    plot_data = df[['market_professionalization_score', 'total_listings']].dropna()
    for idx, row in plot_data.iterrows():
        city = df.loc[idx, 'city']
        ax.scatter(row['total_listings'], row['market_professionalization_score'], 
                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(city, (row['total_listings'], row['market_professionalization_score']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold', alpha=0.8)
    add_quadrant_labels_metrics(ax, plot_data, 'total_listings', 'market_professionalization_score', 'Market Size', 'Professionalization')
    ax.set_xlabel('Total Listings', fontweight='bold', fontsize=11)
    ax.set_ylabel('Professionalization Score', fontweight='bold', fontsize=11)
    ax.set_title('Market Size vs Professionalization', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Median Log Price vs Median ROI
    ax = axes2[1, 1]
    if 'median_log_price' in df.columns:
        plot_data = df[['median_log_price', 'median_roi']].dropna()
        for idx, row in plot_data.iterrows():
            city = df.loc[idx, 'city']
            ax.scatter(row['median_log_price'], row['median_roi'], 
                      s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.annotate(city, (row['median_log_price'], row['median_roi']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.8)
        add_quadrant_labels_metrics(ax, plot_data, 'median_log_price', 'median_roi', 'Log Price', 'ROI')
    ax.set_xlabel('Median Log Price', fontweight='bold', fontsize=11)
    ax.set_ylabel('Median ROI (%)', fontweight='bold', fontsize=11)
    ax.set_title('Log Price vs ROI', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = output_dir / 'city_metrics_comparison.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file2}")
    plt.close()
    
    # Save data to CSV
    csv_file = output_dir / 'city_comparison_data.csv'
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    return df


def main():
    """Main execution function"""
    use_detailed = '-all' in sys.argv
    
    # Discover cities automatically
    base_path = Path('.')
    exclude_dirs = {'__pycache__', 'Census', 'Kaggle', 'Zillow', 'old_scripts', 
                    'city_comparison_outputs', '.git'}
    
    city_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in exclude_dirs:
            if (item / 'listings.csv').exists() or (item / 'listings.csv.gz').exists():
                city_folders.append(item.name)
    
    city_folders = sorted(city_folders)
    
    # Filter out Paris if present
    if 'Paris' in city_folders:
        city_folders.remove('Paris')
    
    print(f"\n{'='*80}")
    print(f"CITY COMPARISON SCATTER PLOTS")
    print(f"{'='*80}")
    print(f"\nFound {len(city_folders)} cities: {', '.join(city_folders)}")
    
    if use_detailed:
        print(f"\n🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"\n🔍 MODE: SIMPLE ANALYSIS (19 variables)")
    
    # Load census data for city size
    print(f"\nLoading census data...")
    try:
        census_df = load_census_data(base_dir='.')
    except Exception as e:
        print(f"  ⚠️  Could not load census data: {e}")
        census_df = None
    
    # Collect data for all cities
    city_data = []
    
    for city_name in city_folders:
        print(f"\nProcessing {city_name}...")
        try:
            df = load_city_data(city_name, base_dir='.', use_detailed=use_detailed)
            if df is None:
                print(f"  ⚠️  Could not load data for {city_name}")
                continue
            
            # Calculate metrics
            metrics = calculate_city_metrics(df, city_name)
            
            # Calculate regression coefficients
            reg_coeffs = calculate_regression_coefficients(df, city_name)
            if reg_coeffs:
                metrics.update(reg_coeffs)
            
            # Add census data if available
            if census_df is not None:
                census_match = match_city_to_census(city_name, census_df)
                if census_match is not None:
                    metrics['population_2024'] = census_match.get('population_2024', np.nan)
            
            city_data.append(metrics)
            print(f"  ✓ Processed {city_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not city_data:
        print("\n❌ No city data collected!")
        return
    
    # Create visualizations
    df_results = create_city_comparison_plots(city_data)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nSuccessfully processed {len(city_data)} cities")
    print(f"\nOutput files saved to: city_comparison_outputs/")
    print(f"  - city_regression_coefficients_comparison.png")
    print(f"  - city_metrics_comparison.png")
    print(f"  - city_comparison_data.csv")


if __name__ == "__main__":
    main()

