"""
Helper utility functions for data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import constants from visualization module
from visualization.styles import MAX_TEXT_LENGTH, TOP_CORRELATIONS_N


def sanitize_text(text, max_length=MAX_TEXT_LENGTH):
    """
    Sanitize text for safe CSV storage
    - Truncate to max_length
    - Remove newlines and extra spaces
    - Remove problematic characters
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of text
    
    Returns:
        Sanitized text string
    """
    if pd.isna(text) or text is None:
        return text
    
    text = str(text)
    
    # Remove newlines and tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove <br /> tags
    text = text.replace('<br />', ' ').replace('<br>', ' ')
    
    # Collapse multiple spaces
    text = ' '.join(text.split())
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + '...'
    
    return text


def analyze_variable(series, var_name):
    """
    Comprehensive analysis of a single variable
    
    Returns dict with: name, type, range, min, max, mean, median, std, precision,
    and for discrete: mode, n_unique
    
    Args:
        series: pandas Series to analyze
        var_name: Name of the variable
    
    Returns:
        Dictionary with variable statistics
    """
    stats = {
        'variable_name': var_name,
        'data_type': None,
        'range': None,
        'min': None,
        'max': None,
        'mean': None,
        'median': None,
        'std': None,
        'precision': None,
        'mode': None,
        'n_unique': None,
        'n_missing': series.isna().sum(),
        'pct_missing': (series.isna().sum() / len(series) * 100).round(2)
    }
    
    # Remove missing values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        stats['data_type'] = 'empty'
        return stats
    
    # Determine if continuous or discrete
    if pd.api.types.is_numeric_dtype(clean_series):
        n_unique = clean_series.nunique()
        
        # Heuristic: if fewer than 20 unique values or all integers, treat as discrete
        if n_unique < 20 or (clean_series.dtype in ['int64', 'int32'] and n_unique < 50):
            stats['data_type'] = 'discrete_numeric'
        else:
            stats['data_type'] = 'continuous'
        
        # Calculate statistics
        stats['min'] = clean_series.min()
        stats['max'] = clean_series.max()
        stats['range'] = stats['max'] - stats['min']
        stats['mean'] = clean_series.mean()
        stats['median'] = clean_series.median()
        stats['std'] = clean_series.std()
        
        # Precision (number of decimal places)
        if stats['data_type'] == 'continuous':
            # Sample some values to determine precision
            sample_vals = clean_series.head(100).astype(str)
            decimal_places = [len(str(v).split('.')[-1]) if '.' in str(v) else 0 for v in sample_vals]
            stats['precision'] = max(decimal_places) if decimal_places else 0
        else:
            stats['precision'] = 0
        
        # Mode and unique count
        stats['mode'] = clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else None
        stats['n_unique'] = n_unique
        
    else:
        # Categorical/text variable
        stats['data_type'] = 'discrete_categorical'
        stats['n_unique'] = clean_series.nunique()
        
        # Sanitize text values for min, max, mode
        mode_val = clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else None
        stats['mode'] = sanitize_text(mode_val, max_length=50)
        
        # For categorical, min/max are first/last alphabetically
        stats['min'] = sanitize_text(clean_series.min(), max_length=50)
        stats['max'] = sanitize_text(clean_series.max(), max_length=50)
    
    return stats


def create_variable_summary_table(df, city_name):
    """
    Create comprehensive variable summary table for a city
    
    Args:
        df: DataFrame with city data
        city_name: Name of the city
    
    Returns:
        DataFrame with variable statistics
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING VARIABLES FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Analyze each variable
    all_stats = []
    for col in df.columns:
        if col != 'city':  # Skip city identifier
            print(f"  Analyzing: {col}")
            stats = analyze_variable(df[col], col)
            all_stats.append(stats)
    
    # Create DataFrame
    summary_df = pd.DataFrame(all_stats)
    
    # Reorder columns for better readability
    col_order = ['variable_name', 'data_type', 'n_unique', 'n_missing', 'pct_missing',
                 'min', 'max', 'range', 'mean', 'median', 'std', 'precision', 'mode']
    summary_df = summary_df[col_order]
    
    return summary_df


def create_all_correlation_matrices(df, city_name, output_dir, top_n=TOP_CORRELATIONS_N, create_scatter_plots=False):
    """
    Create correlation matrices for all combinations of numeric variables
    
    Args:
        df: DataFrame with city data
        city_name: Name of the city
        output_dir: Directory to save outputs
        top_n: Number of top correlations to analyze (default 25)
        create_scatter_plots: Whether to create scatter plots for top correlations
    """
    print(f"\n{'='*80}")
    print(f"CREATING CORRELATION MATRICES FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'city']
    
    if len(numeric_cols) < 2:
        print(f"  [WARNING]  Not enough numeric variables for correlation analysis")
        return
    
    print(f"  Found {len(numeric_cols)} numeric variables")
    print(f"  Variables: {', '.join(numeric_cols)}")
    
    # Create correlation matrix for all numeric variables
    print(f"\n  Creating full correlation matrix...")
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix as CSV
    corr_matrix.to_csv(output_dir / f'{city_name}_correlation_matrix.csv')
    print(f"  [OK] Saved correlation matrix CSV")
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'{city_name.upper()} - Correlation Matrix (All Variables)', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_correlation_heatmap_full.png', 
               dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved full correlation heatmap")
    plt.close()
    
    # Get top correlations
    print(f"\n  Extracting top {top_n} correlations...")
    
    # Get upper triangle of correlation matrix (to avoid duplicates)
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Stack and sort
    corr_pairs = corr_upper.stack().sort_values(ascending=False)
    
    # Get top N positive and top N negative correlations
    top_positive = corr_pairs.head(top_n)
    top_negative = corr_pairs.tail(top_n)
    
    # Combine and create DataFrame
    top_corrs = pd.concat([top_positive, top_negative])
    top_corrs_df = top_corrs.reset_index()
    top_corrs_df.columns = ['variable_1', 'variable_2', 'correlation']
    top_corrs_df = top_corrs_df.sort_values('correlation', ascending=False)
    
    # Save top correlations
    top_corrs_df.to_csv(output_dir / f'{city_name}_top_correlations.csv', index=False)
    print(f"  [OK] Saved top {len(top_corrs_df)} correlations")
    
    # Create scatter plots for top correlations if requested
    if create_scatter_plots and len(top_corrs_df) > 0:
        print(f"\n  Creating scatter plots for top correlations...")
        n_plots = min(6, len(top_corrs_df))  # Limit to 6 plots
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_corrs_df.head(n_plots).iterrows()):
            var1 = row['variable_1']
            var2 = row['variable_2']
            corr_val = row['correlation']
            
            ax = axes[idx]
            ax.scatter(df[var1], df[var2], alpha=0.5, s=20)
            ax.set_xlabel(var1, fontsize=10)
            ax.set_ylabel(var2, fontsize=10)
            ax.set_title(f'{var1} vs {var2}\n(r={corr_val:.3f})', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{city_name}_correlation_scatter_plots.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved scatter plots")
        plt.close()

