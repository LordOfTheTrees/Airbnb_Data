"""
City Data Exploration Script
Creates targeted scatter plots for business case analysis

This script focuses on exploring relationships between property size metrics
and price proxies for a single city.

Usage:
    python explore_city_data.py Austin           # Austin, simple dataset
    python explore_city_data.py Austin -all      # Austin, detailed dataset
    python explore_city_data.py Chicago          # Chicago, simple dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import io
from scipy import stats

# Set UTF-8 encoding for Windows compatibility
# Only wrap stdout/stderr when running as main script, not when imported
if sys.platform == 'win32' and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Import feature engineering functions from city_level_analysis
from city_level_analysis import (
    apply_all_feature_engineering,
    add_log_transforms,
    add_within_city_metrics,
    add_revenue_proxies
)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_city_data(city_name, base_dir='.', use_detailed=False):
    """
    Load and prepare data for a single city
    
    Args:
        city_name: Name of the city folder
        base_dir: Base directory containing city folders
        use_detailed: If True, use detailed 79-variable dataset
    
    Returns:
        DataFrame with feature engineering applied
    """
    city_path = Path(base_dir) / city_name
    
    if not city_path.exists():
        raise FileNotFoundError(f"City directory not found: {city_path}")
    
    # Determine which file to use
    if use_detailed:
        listings_file = city_path / 'listings.csv.gz'
        if not listings_file.exists():
            print(f"WARNING: Detailed file not found, trying simple file...")
            listings_file = city_path / 'listings.csv'
    else:
        listings_file = city_path / 'listings.csv'
        if not listings_file.exists():
            print(f"WARNING: Simple file not found, trying detailed file...")
            listings_file = city_path / 'listings.csv.gz'
    
    if not listings_file.exists():
        raise FileNotFoundError(f"No listings file found for {city_name}")
    
    print(f"Loading data from: {listings_file}")
    
    # Load data
    if str(listings_file).endswith('.gz'):
        df = pd.read_csv(listings_file, compression='gzip')
    else:
        df = pd.read_csv(listings_file)
    
    print(f"Loaded {len(df):,} listings with {len(df.columns)} columns")
    
    # Clean price if present
    if 'price' in df.columns:
        df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    
    # Apply feature engineering
    df = apply_all_feature_engineering(df, city_name)
    
    return df


def create_size_vs_price_scatter_plots(df, city_name, output_dir=None, price_vars=None):
    """
    Create scatter plots: size metrics vs price proxies
    
    X-axis variables: accommodates, bathrooms, bedrooms, beds
    Y-axis variables: specified price variables (default: log_price only)
    
    Args:
        df: DataFrame with feature engineering applied
        city_name: Name of city (for titles/filenames)
        output_dir: Directory to save plots (if None, saves to current directory)
        price_vars: List of price variables to plot (default: ['log_price'])
    """
    if price_vars is None:
        price_vars = ['log_price']
    
    print(f"\n{'='*80}")
    print(f"CREATING SIZE vs PRICE SCATTER PLOTS FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Define variables
    size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    
    # Check which variables are available
    available_size = [v for v in size_vars if v in df.columns]
    available_price = [v for v in price_vars if v in df.columns]
    
    if not available_size:
        print("WARNING: No size variables found in data")
        return
    
    if not available_price:
        print("WARNING: No price variables found in data")
        return
    
    print(f"\nSize variables: {', '.join(available_size)}")
    print(f"Price variables: {', '.join(available_price)}")
    
    # Filter out missing values for cleaner plots
    plot_vars = available_size + available_price
    df_plot = df[plot_vars].dropna()
    
    # Filter outliers for log_price scatter plots
    # Remove listings with: bathrooms > 8, bedrooms > 10, beds > 20
    if 'log_price' in available_price:
        initial_count = len(df_plot)
        if 'bathrooms' in df_plot.columns:
            df_plot = df_plot[df_plot['bathrooms'] <= 8]
        if 'bedrooms' in df_plot.columns:
            df_plot = df_plot[df_plot['bedrooms'] <= 10]
        if 'beds' in df_plot.columns:
            df_plot = df_plot[df_plot['beds'] <= 20]
        filtered_count = len(df_plot)
        if initial_count != filtered_count:
            print(f"  Filtered outliers: {initial_count:,} → {filtered_count:,} listings")
            print(f"    (Removed: bathrooms > 8, bedrooms > 10, beds > 20)")
    
    print(f"\nUsing {len(df_plot):,} listings with complete data")
    
    # Create figure with subplots: 2x2 quad chart for ROI workflow (when 4 size vars and 1 price var)
    # Otherwise use flexible layout
    if len(available_size) == 4 and len(available_price) == 1:
        # Quad chart format for ROI workflow
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        # Flexible layout for other cases
        fig, axes = plt.subplots(len(available_size), len(available_price), 
                                figsize=(18, 5 * len(available_size)))
    
    # Handle case where we only have one row or one column (only if not quad chart)
    if len(available_size) == 4 and len(available_price) == 1:
        # Already flattened for quad chart
        pass
    else:
        if len(available_size) == 1:
            axes = axes.reshape(1, -1)
        if len(available_price) == 1:
            axes = axes.reshape(-1, 1)
    
    # Store regression results for documentation
    regression_results = []
    
    # Create scatter plots
    plot_idx = 0
    for i, size_var in enumerate(available_size):
        for j, price_var in enumerate(available_price):
            # Handle quad chart layout (flattened) vs flexible layout
            if len(available_size) == 4 and len(available_price) == 1:
                ax = axes[plot_idx]
                plot_idx += 1
            else:
                ax = axes[i, j]
            
            # Get data for this specific plot
            plot_data = df_plot[[size_var, price_var]].dropna()
            
            if len(plot_data) == 0:
                continue
            
            # Create scatter plot
            ax.scatter(plot_data[size_var], plot_data[price_var], 
                      alpha=0.4, s=30, edgecolors='none')
            
            # Calculate linear regression
            if pd.api.types.is_numeric_dtype(plot_data[size_var]) and \
               pd.api.types.is_numeric_dtype(plot_data[price_var]):
                x = plot_data[size_var].values
                y = plot_data[price_var].values
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Generate regression line
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit', alpha=0.8)
                
                # Calculate correlation
                corr = plot_data[size_var].corr(plot_data[price_var])
                
                # Format equation
                if intercept >= 0:
                    eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
                else:
                    eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
                
                # Add title with correlation and equation
                ax.set_title(f'r = {corr:.3f}\n{eq_text}', 
                           fontsize=11, fontweight='bold')
                
                # Store results for documentation
                regression_results.append({
                    'city': city_name,
                    'size_var': size_var,
                    'price_var': price_var,
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'n': len(plot_data)
                })
            
            # Add labels
            ax.set_xlabel(size_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=11)
            ax.set_ylabel(price_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=11)
            
            # Add grid
            ax.grid(True, alpha=0.3)
    
    # Add overall title
    price_vars_str = ', '.join([v.replace('_', ' ').title() for v in available_price])
    fig.suptitle(f'{city_name.upper()} - Property Size vs Price ({price_vars_str})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Create filename based on price vars
    # For ROI workflow, we want a specific quad chart format
    if len(available_price) == 1 and len(available_size) == 4:
        # Create 2x2 quad chart for ROI workflow
        filename = f'{city_name}_size_vs_{available_price[0]}.png'
    else:
        filename = f'{city_name}_size_vs_price_scatter.png'
    
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # Also create individual plots for easier inspection
    print(f"\nCreating individual plots...")
    individual_dir = output_dir / f'{city_name}_individual_plots'
    individual_dir.mkdir(exist_ok=True)
    
    for size_var in available_size:
        for price_var in available_price:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Filter data (apply same outlier filtering)
            plot_data = df_plot[[size_var, price_var]].dropna()
            
            if len(plot_data) > 0:
                ax.scatter(plot_data[size_var], plot_data[price_var], 
                          alpha=0.5, s=40, edgecolors='none')
                
                # Calculate linear regression
                if pd.api.types.is_numeric_dtype(plot_data[size_var]) and \
                   pd.api.types.is_numeric_dtype(plot_data[price_var]):
                    x = plot_data[size_var].values
                    y = plot_data[price_var].values
                    
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Generate regression line
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit', alpha=0.8)
                    
                    # Calculate correlation
                    corr = plot_data[size_var].corr(plot_data[price_var])
                    
                    # Format equation
                    if intercept >= 0:
                        eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
                    else:
                        eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
                    
                    ax.set_title(f'{city_name.upper()}: {size_var} vs {price_var}\n'
                               f'Correlation: r = {corr:.3f} | {eq_text}', 
                               fontsize=12, fontweight='bold')
                
                ax.set_xlabel(size_var.replace('_', ' ').title(), 
                             fontweight='bold', fontsize=12)
                ax.set_ylabel(price_var.replace('_', ' ').title(), 
                             fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                filename = f'{city_name}_{size_var}_vs_{price_var}.png'
                plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"Saved {len(available_size) * len(available_price)} individual plots to: {individual_dir}/")
    
    # Print regression results
    if regression_results:
        print(f"\n{'='*80}")
        print(f"LINEAR REGRESSION RESULTS")
        print(f"{'='*80}")
        for result in regression_results:
            if result['price_var'] == 'log_price':  # Only print for log_price
                print(f"\n{result['size_var'].replace('_', ' ').title()} vs {result['price_var'].replace('_', ' ').title()}:")
                if result['intercept'] >= 0:
                    print(f"  Equation: y = {result['slope']:.4f}x + {result['intercept']:.4f}")
                else:
                    print(f"  Equation: y = {result['slope']:.4f}x - {abs(result['intercept']):.4f}")
                print(f"  R² = {result['r_squared']:.4f} (r = {result['r_value']:.4f})")
                print(f"  p-value = {result['p_value']:.2e}")
                print(f"  n = {result['n']:,}")
                print(f"  Interpretation: Each additional {result['size_var'].replace('_', ' ')} increases log(price) by {result['slope']:.4f}")
    
    # Save regression results to CSV for documentation
    if regression_results:
        regression_df = pd.DataFrame(regression_results)
        regression_file = output_dir / f'{city_name}_linear_regression_results.csv'
        regression_df.to_csv(regression_file, index=False)
        print(f"\n✓ Saved regression results to: {regression_file}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for size_var in available_size:
        if size_var in df.columns:
            print(f"\n{size_var.replace('_', ' ').title()}:")
            print(f"  Mean: {df[size_var].mean():.2f}")
            print(f"  Median: {df[size_var].median():.2f}")
            print(f"  Range: {df[size_var].min():.0f} - {df[size_var].max():.0f}")
    
    for price_var in available_price:
        if price_var in df.columns:
            print(f"\n{price_var.replace('_', ' ').title()}:")
            print(f"  Mean: {df[price_var].mean():.2f}")
            print(f"  Median: {df[price_var].median():.2f}")
            if price_var != 'price_percentile':  # Percentile is always 0-1
                print(f"  Range: {df[price_var].min():.2f} - {df[price_var].max():.2f}")


def create_size_vs_occupancy_scatter_plots(df, city_name, output_dir=None):
    """
    Create scatter plots: size metrics vs occupancy metrics
    
    X-axis variables: accommodates, bathrooms, bedrooms, beds
    Y-axis variable: occupancy_rate (from estimated_occupancy_l365d, actual booked days)
    
    Creates a 4x1 grid (4 plots total)
    
    Args:
        df: DataFrame with feature engineering applied
        city_name: Name of city (for titles/filenames)
        output_dir: Directory to save plots (if None, saves to current directory)
    """
    print(f"\n{'='*80}")
    print(f"CREATING SIZE vs OCCUPANCY SCATTER PLOTS FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Define variables
    size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    # Prefer occupancy_rate (from estimated_occupancy_l365d), fallback to calendar_unavailable_proxy
    if 'occupancy_rate' in df.columns:
        occupancy_var = 'occupancy_rate'
    elif 'calendar_unavailable_proxy' in df.columns:
        occupancy_var = 'calendar_unavailable_proxy'
        print("WARNING: Using calendar_unavailable_proxy (less accurate than occupancy_rate)")
    else:
        print("WARNING: No occupancy metrics found in data")
        print("Make sure feature engineering has been applied")
        return
    
    # Check which variables are available
    available_size = [v for v in size_vars if v in df.columns]
    
    if occupancy_var not in df.columns:
        print(f"WARNING: {occupancy_var} not found in data")
        return
    
    if not available_size:
        print("WARNING: No size variables found in data")
        return
    
    print(f"\nSize variables: {', '.join(available_size)}")
    print(f"Occupancy variable: {occupancy_var}")
    
    # Filter out missing values for cleaner plots
    plot_vars = available_size + [occupancy_var]
    df_plot = df[plot_vars].dropna()
    
    print(f"\nUsing {len(df_plot):,} listings with complete data")
    
    # Create figure with subplots: 4 rows (size vars) × 1 col (occupancy)
    fig, axes = plt.subplots(len(available_size), 1, 
                            figsize=(10, 4 * len(available_size)))
    
    # Handle case where we only have one row
    if len(available_size) == 1:
        axes = [axes]
    
    # Create scatter plots
    for i, size_var in enumerate(available_size):
        ax = axes[i]
        
        # Create scatter plot
        ax.scatter(df_plot[size_var], df_plot[occupancy_var], 
                  alpha=0.4, s=30, edgecolors='none')
        
        # Add labels
        ax.set_xlabel(size_var.replace('_', ' ').title(), 
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('Occupancy Rate (0-1)', 
                     fontweight='bold', fontsize=12)
        
        # Calculate linear regression if both are numeric
        if pd.api.types.is_numeric_dtype(df_plot[size_var]) and \
           pd.api.types.is_numeric_dtype(df_plot[occupancy_var]):
            x = df_plot[size_var].values
            y = df_plot[occupancy_var].values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Generate regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit', alpha=0.8)
            
            # Calculate correlation
            corr = df_plot[size_var].corr(df_plot[occupancy_var])
            
            # Format equation
            if intercept >= 0:
                eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
            else:
                eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
            
            ax.set_title(f'{size_var.replace("_", " ").title()} vs Occupancy\n'
                        f'Correlation: r = {corr:.3f} | {eq_text} | R² = {r_value**2:.4f}', 
                        fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for occupancy (0-1)
        ax.set_ylim(-0.05, 1.05)
    
    # Add overall title
    fig.suptitle(f'{city_name.upper()} - Property Size vs Projected Average Yearly Occupancy', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'{city_name}_size_vs_occupancy_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # Also create individual plots for easier inspection
    print(f"\nCreating individual plots...")
    individual_dir = output_dir / f'{city_name}_occupancy_individual_plots'
    individual_dir.mkdir(exist_ok=True)
    
    for size_var in available_size:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Filter data
        plot_data = df[[size_var, occupancy_var]].dropna()
        
        if len(plot_data) > 0:
            ax.scatter(plot_data[size_var], plot_data[occupancy_var], 
                      alpha=0.5, s=40, edgecolors='none')
            
            # Calculate linear regression
            if pd.api.types.is_numeric_dtype(plot_data[size_var]) and \
               pd.api.types.is_numeric_dtype(plot_data[occupancy_var]):
                x = plot_data[size_var].values
                y = plot_data[occupancy_var].values
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Generate regression line
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit', alpha=0.8)
                
                # Calculate correlation
                corr = plot_data[size_var].corr(plot_data[occupancy_var])
                
                # Format equation
                if intercept >= 0:
                    eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
                else:
                    eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
                
                ax.set_title(f'{city_name.upper()}: {size_var} vs Occupancy\n'
                           f'Correlation: r = {corr:.3f} | {eq_text} | R² = {r_value**2:.4f}', 
                           fontsize=12, fontweight='bold')
            
            ax.set_xlabel(size_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=12)
            ax.set_ylabel('Occupancy Rate (0-1)', 
                         fontweight='bold', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'{city_name}_{size_var}_vs_occupancy.png'
            plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Saved {len(available_size)} individual plots to: {individual_dir}/")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for size_var in available_size:
        if size_var in df.columns:
            print(f"\n{size_var.replace('_', ' ').title()}:")
            print(f"  Mean: {df[size_var].mean():.2f}")
            print(f"  Median: {df[size_var].median():.2f}")
            print(f"  Range: {df[size_var].min():.0f} - {df[size_var].max():.0f}")
    
    if occupancy_var in df.columns:
        print(f"\n{occupancy_var.replace('_', ' ').title()}:")
        print(f"  Mean: {df[occupancy_var].mean():.3f}")
        print(f"  Median: {df[occupancy_var].median():.3f}")
        print(f"  Range: {df[occupancy_var].min():.3f} - {df[occupancy_var].max():.3f}")
        if occupancy_var == 'occupancy_rate':
            print(f"  Note: Occupancy rate uses estimated_occupancy_l365d (actual booked days)")
        else:
            print(f"  Note: Using calendar_unavailable_proxy (includes host-blocked days)")


def create_size_vs_occupancy_boxplots(df, city_name, output_dir=None, use_log=False):
    """
    Create box plots: size metrics vs occupancy (binned categories)
    
    X-axis: Binned size categories (accommodates, bathrooms, bedrooms, beds)
    Y-axis: occupancy_rate (from estimated_occupancy_l365d)
    
    Creates box plots showing distribution of occupancy by size category
    
    Args:
        df: DataFrame with feature engineering applied
        city_name: Name of city (for titles/filenames)
        output_dir: Directory to save plots
        use_log: If True, use log-transformed size variables for binning
    """
    print(f"\n{'='*80}")
    print(f"CREATING SIZE vs OCCUPANCY BOX PLOTS FOR {city_name.upper()}")
    if use_log:
        print("Using log-transformed size variables")
    print(f"{'='*80}")
    
    # Define variables
    size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    # Prefer occupancy_rate (from estimated_occupancy_l365d), fallback to calendar_unavailable_proxy
    if 'occupancy_rate' in df.columns:
        occupancy_var = 'occupancy_rate'
    elif 'calendar_unavailable_proxy' in df.columns:
        occupancy_var = 'calendar_unavailable_proxy'
        print("WARNING: Using calendar_unavailable_proxy (less accurate than occupancy_rate)")
    else:
        print("WARNING: No occupancy metrics found in data")
        return
    
    # Check which variables are available
    available_size = [v for v in size_vars if v in df.columns]
    
    if occupancy_var not in df.columns:
        print(f"WARNING: {occupancy_var} not found in data")
        return
    
    if not available_size:
        print("WARNING: No size variables found in data")
        return
    
    print(f"\nSize variables: {', '.join(available_size)}")
    
    # Filter out missing values
    plot_vars = available_size + [occupancy_var]
    df_plot = df[plot_vars].dropna().copy()
    
    print(f"\nUsing {len(df_plot):,} listings with complete data")
    
    # Create figure with subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define binning strategy for each variable
    binning_strategies = {
        'accommodates': {
            'bins': [0, 2, 4, 6, 8, 20],
            'labels': ['1-2', '3-4', '5-6', '7-8', '9+'],
        },
        'bathrooms': {
            'bins': [0, 1, 2, 3, 4, 20],
            'labels': ['1', '1.5-2', '2.5-3', '3.5-4', '4+'],
        },
        'bedrooms': {
            'bins': [-0.5, 0.5, 1.5, 2.5, 3.5, 20],
            'labels': ['Studio', '1BR', '2BR', '3BR', '4+'],
        },
        'beds': {
            'bins': [0, 2, 4, 6, 8, 20],
            'labels': ['1-2', '3-4', '5-6', '7-8', '9+'],
        }
    }
    
    # Create box plots
    for i, size_var in enumerate(available_size):
        if i >= 4:  # Only plot first 4
            break
            
        ax = axes[i]
        
        # Determine which variable to use and binning strategy
        if use_log:
            # Use log-transformed variables if available, but bin in log space and convert back to original scale for labels
            if size_var == 'accommodates' and 'log_accommodates' in df_plot.columns:
                var_to_use = 'log_accommodates'
                # Create bins in log space using quantiles
                log_values = df_plot[var_to_use].dropna()
                if len(log_values) > 0:
                    log_bins = [log_values.min()] + list(log_values.quantile([0.2, 0.4, 0.6, 0.8])) + [log_values.max()]
                    # Convert log bin edges back to original scale for labels
                    original_bins = [np.exp(b) for b in log_bins]
                    # Create meaningful labels showing the range in original scale
                    labels = []
                    for j in range(len(original_bins)-1):
                        low = int(np.round(original_bins[j]))
                        high = int(np.round(original_bins[j+1]))
                        if j == 0:
                            labels.append(f'{low}-{high}')
                        elif j == len(original_bins)-2:
                            labels.append(f'{low}+')
                        else:
                            labels.append(f'{low}-{high}')
                    bins = log_bins
                else:
                    var_to_use = size_var
                    bins = binning_strategies[size_var]['bins']
                    labels = binning_strategies[size_var]['labels']
            elif size_var == 'beds' and 'log_beds' in df_plot.columns:
                var_to_use = 'log_beds'
                log_values = df_plot[var_to_use].dropna()
                if len(log_values) > 0:
                    log_bins = [log_values.min()] + list(log_values.quantile([0.2, 0.4, 0.6, 0.8])) + [log_values.max()]
                    # Convert log bin edges back to original scale for labels
                    original_bins = [np.exp(b) - 1 for b in log_bins]  # log_beds uses log1p, so reverse with exp-1
                    labels = []
                    for j in range(len(original_bins)-1):
                        low = int(np.round(original_bins[j]))
                        high = int(np.round(original_bins[j+1]))
                        if j == 0:
                            labels.append(f'{low}-{high}')
                        elif j == len(original_bins)-2:
                            labels.append(f'{low}+')
                        else:
                            labels.append(f'{low}-{high}')
                    bins = log_bins
                else:
                    var_to_use = size_var
                    bins = binning_strategies[size_var]['bins']
                    labels = binning_strategies[size_var]['labels']
            else:
                # For bathrooms and bedrooms, no log transform available, so skip log version
                # (they would be identical to non-log version)
                var_to_use = size_var
                bins = binning_strategies[size_var]['bins']
                labels = binning_strategies[size_var]['labels']
        else:
            var_to_use = size_var
            bins = binning_strategies[size_var]['bins']
            labels = binning_strategies[size_var]['labels']
        
        # Create bins
        if var_to_use in df_plot.columns:
            df_plot['binned'] = pd.cut(df_plot[var_to_use], bins=bins, labels=labels, include_lowest=True)
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            for label in labels:
                subset = df_plot[df_plot['binned'] == label][occupancy_var]
                if len(subset) > 0:
                    box_data.append(subset.values)
                    box_labels.append(label)
            
            if len(box_data) > 0:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{size_var.replace("_", " ").title()} vs Occupancy', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Size Category', fontweight='bold', fontsize=11)
                ax.set_ylabel('Occupancy Rate (0-1)', fontweight='bold', fontsize=11)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Calculate and print median occupancy by bin
                medians = df_plot.groupby('binned')[occupancy_var].median()
                print(f"\n{size_var} - Median Occupancy by Category:")
                for cat, med in medians.items():
                    count = len(df_plot[df_plot['binned'] == cat])
                    print(f"  {cat}: {med:.3f} (n={count:,})")
    
    # Add overall title
    transform_text = " (Log-Transformed)" if use_log else ""
    fig.suptitle(f'{city_name.upper()} - Occupancy Distribution by Size Category{transform_text}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    suffix = '_log' if use_log else ''
    output_file = output_dir / f'{city_name}_size_vs_occupancy_boxplots{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def create_size_vs_occupancy_hexbin(df, city_name, output_dir=None):
    """
    Create hexbin density plots: size metrics vs occupancy
    
    Shows density of data points using hexagonal binning
    Uses occupancy_rate (from estimated_occupancy_l365d) when available
    
    Args:
        df: DataFrame with feature engineering applied
        city_name: Name of city (for titles/filenames)
        output_dir: Directory to save plots
    """
    print(f"\n{'='*80}")
    print(f"CREATING SIZE vs OCCUPANCY HEXBIN DENSITY PLOTS FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Define variables
    size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    # Prefer occupancy_rate (from estimated_occupancy_l365d), fallback to calendar_unavailable_proxy
    if 'occupancy_rate' in df.columns:
        occupancy_var = 'occupancy_rate'
    elif 'calendar_unavailable_proxy' in df.columns:
        occupancy_var = 'calendar_unavailable_proxy'
        print("WARNING: Using calendar_unavailable_proxy (less accurate than occupancy_rate)")
    else:
        print("WARNING: No occupancy metrics found in data")
        return
    
    # Check which variables are available
    available_size = [v for v in size_vars if v in df.columns]
    
    if occupancy_var not in df.columns:
        print(f"WARNING: {occupancy_var} not found in data")
        return
    
    if not available_size:
        print("WARNING: No size variables found in data")
        return
    
    print(f"\nSize variables: {', '.join(available_size)}")
    
    # Filter out missing values
    plot_vars = available_size + [occupancy_var]
    df_plot = df[plot_vars].dropna()
    
    print(f"\nUsing {len(df_plot):,} listings with complete data")
    
    # Create figure with subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create hexbin plots
    for i, size_var in enumerate(available_size):
        if i >= 4:  # Only plot first 4
            break
            
        ax = axes[i]
        
        # Filter to reasonable range (95th percentile) to focus on main data
        p95 = df_plot[size_var].quantile(0.95)
        df_filtered = df_plot[df_plot[size_var] <= p95].copy()
        
        # Create hexbin plot
        hb = ax.hexbin(df_filtered[size_var], df_filtered[occupancy_var], 
                      gridsize=20, cmap='YlOrRd', mincnt=1)
        
        ax.set_xlabel(size_var.replace('_', ' ').title(), 
                     fontweight='bold', fontsize=12)
        ax.set_ylabel('Occupancy Rate (0-1)', 
                     fontweight='bold', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{size_var.replace("_", " ").title()} vs Occupancy (Density)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'{city_name.upper()} - Occupancy Density by Property Size (Hexbin)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'{city_name}_size_vs_occupancy_hexbin.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # Also create individual hexbin plots
    print(f"\nCreating individual hexbin plots...")
    individual_dir = output_dir / f'{city_name}_occupancy_hexbin_individual'
    individual_dir.mkdir(exist_ok=True)
    
    for size_var in available_size:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Filter data
        plot_data = df[[size_var, occupancy_var]].dropna()
        p95 = plot_data[size_var].quantile(0.95)
        plot_data = plot_data[plot_data[size_var] <= p95]
        
        if len(plot_data) > 0:
            hb = ax.hexbin(plot_data[size_var], plot_data[occupancy_var], 
                          gridsize=25, cmap='YlOrRd', mincnt=1)
            
            ax.set_xlabel(size_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=12)
            ax.set_ylabel('Occupancy Rate (0-1)', 
                         fontweight='bold', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{city_name.upper()}: {size_var} vs Occupancy (Density)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Count', fontweight='bold')
            
            plt.tight_layout()
            
            filename = f'{city_name}_{size_var}_vs_occupancy_hexbin.png'
            plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Saved {len(available_size)} individual hexbin plots to: {individual_dir}/")


def main():
    """Main execution function"""
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    explore_mode = 'price'  # default
    price_vars = None  # default to log_price only
    
    # Check for exploration mode flags
    if '-occupancy' in sys.argv or '--occupancy' in sys.argv:
        explore_mode = 'occupancy'
    elif '-all-prices' in sys.argv or '--all-prices' in sys.argv:
        explore_mode = 'price'
        price_vars = ['price_clean', 'log_price', 'price_percentile']
    elif '-price' in sys.argv or '--price' in sys.argv:
        explore_mode = 'price'
    # Check if specific price vars requested
    if '-price-vars' in sys.argv:
        idx = sys.argv.index('-price-vars')
        if idx + 1 < len(sys.argv):
            price_vars = sys.argv[idx + 1].split(',')
    
    # Get city name (first non-flag argument)
    flags = ['-all', '-occupancy', '--occupancy', '-all-prices', '--all-prices', 
             '-price', '--price', '-price-vars']
    city_args = [arg for arg in sys.argv[1:] if arg not in flags]
    
    # Also skip the argument after -price-vars if it exists
    if '-price-vars' in sys.argv:
        idx = sys.argv.index('-price-vars')
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in city_args:
            city_args.remove(sys.argv[idx + 1])
    
    if not city_args:
        print("Usage: python explore_city_data.py CityName [options]")
        print("\nOptions:")
        print("  -all              Use detailed dataset (79 variables)")
        print("  -occupancy        Explore size vs occupancy (default: size vs price)")
        print("  -all-prices       Plot all price variables (price_clean, log_price, price_percentile)")
        print("                    (default: log_price only)")
        print("  -price-vars VAR1,VAR2  Specify custom price variables (e.g., price_clean,log_price)")
        print("\nExamples:")
        print("  python explore_city_data.py Austin -all")
        print("  python explore_city_data.py Austin -all -occupancy")
        print("  python explore_city_data.py Austin -all -all-prices")
        print("  python explore_city_data.py Austin -all -price-vars price_clean")
        sys.exit(1)
    
    city_name = city_args[0]
    
    print(f"\n{'#'*80}")
    print(f"CITY DATA EXPLORATION - {city_name.upper()}")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\nMODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"\nMODE: SIMPLE ANALYSIS (19 variables)")
    
    print(f"\nEXPLORATION MODE: {explore_mode.upper()}")
    if explore_mode == 'price':
        if price_vars:
            print(f"Price variables: {', '.join(price_vars)}")
        else:
            print(f"Price variables: log_price (default - use -all-prices or -price-vars for others)")
    
    try:
        # Load data
        df = load_city_data(city_name, base_dir='.', use_detailed=use_detailed)
        
        # Create output directory
        output_dir = Path(city_name) / 'exploration_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scatter plots based on mode
        if explore_mode == 'occupancy':
            # Original scatter plots
            create_size_vs_occupancy_scatter_plots(df, city_name, output_dir=output_dir)
            
            # Box plots (log transform removed - not meaningful for categorical binning)
            # The log transformation is better suited for scatter plots where we can see the continuous relationship
            create_size_vs_occupancy_boxplots(df, city_name, output_dir=output_dir, use_log=False)
            
            # Hexbin density plots
            create_size_vs_occupancy_hexbin(df, city_name, output_dir=output_dir)
            
            print(f"\n{'#'*80}")
            print(f"EXPLORATION COMPLETE FOR {city_name.upper()}")
            print(f"{'#'*80}")
            print(f"\nOutput saved to: {output_dir}/")
            print(f"  - {city_name}_size_vs_occupancy_scatter.png (scatter plots)")
            print(f"  - {city_name}_size_vs_occupancy_boxplots.png (box plots)")
            print(f"  - {city_name}_size_vs_occupancy_hexbin.png (hexbin density plots)")
            print(f"  - {city_name}_occupancy_individual_plots/ (individual scatter plots)")
            print(f"  - {city_name}_occupancy_hexbin_individual/ (individual hexbin plots)")
        else:  # price mode
            create_size_vs_price_scatter_plots(df, city_name, output_dir=output_dir, price_vars=price_vars)
            print(f"\n{'#'*80}")
            print(f"EXPLORATION COMPLETE FOR {city_name.upper()}")
            print(f"{'#'*80}")
            print(f"\nOutput saved to: {output_dir}/")
            if price_vars and len(price_vars) > 1:
                print(f"  - {city_name}_size_vs_price_scatter.png (combined grid)")
                print(f"  - {city_name}_individual_plots/ (individual plots)")
            else:
                print(f"  - {city_name}_size_vs_log_price.png (combined grid)")
                print(f"  - {city_name}_individual_plots/ (individual plots)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

