"""
City-Level Analysis Script
Generates comprehensive statistics and correlation matrices for each city individually
Run this from your main Airbnb_Data directory

Usage:
    python city_level_analysis.py           # Uses simple 19-column listings.csv
    python city_level_analysis.py -all      # Uses detailed 79-column listings.csv.gz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

def analyze_variable(series, var_name):
    """
    Comprehensive analysis of a single variable
    
    Returns dict with: name, type, range, min, max, mean, median, std, precision,
    and for discrete: mode, n_unique
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
        stats['mode'] = clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else None
        
        # For categorical, min/max are first/last alphabetically
        stats['min'] = clean_series.min()
        stats['max'] = clean_series.max()
    
    return stats

def create_variable_summary_table(df, city_name):
    """Create comprehensive variable summary table for a city"""
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

def create_all_correlation_matrices(df, city_name, output_dir):
    """
    Create correlation matrices for all combinations of numeric variables
    """
    print(f"\n{'='*80}")
    print(f"CREATING CORRELATION MATRICES FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'city']
    
    if len(numeric_cols) < 2:
        print(f"  ⚠️  Not enough numeric variables for correlation analysis")
        return
    
    print(f"  Found {len(numeric_cols)} numeric variables")
    print(f"  Variables: {', '.join(numeric_cols)}")
    
    # Create correlation matrix for all numeric variables
    print(f"\n  Creating full correlation matrix...")
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix as CSV
    corr_matrix.to_csv(output_dir / f'{city_name}_correlation_matrix.csv')
    print(f"  ✓ Saved correlation matrix CSV")
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'{city_name.upper()} - Correlation Matrix (All Variables)', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_correlation_heatmap_full.png', 
               dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved full correlation heatmap")
    plt.close()
    
    # Create pairwise correlation plots for key relationships
    print(f"\n  Creating pairwise correlation analysis...")
    
    # Key variable pairs to highlight
    key_vars = ['price_clean', 'number_of_reviews', 'availability_365', 
                'minimum_nights', 'reviews_per_month', 'calculated_host_listings_count']
    available_key_vars = [v for v in key_vars if v in numeric_cols]
    
    if len(available_key_vars) >= 2:
        # Create a detailed correlation report for key variables
        key_corr = corr_matrix.loc[available_key_vars, available_key_vars]
        
        # Find strongest correlations
        print(f"\n  📊 STRONGEST CORRELATIONS:")
        # Get upper triangle of correlation matrix
        upper_tri = key_corr.where(np.triu(np.ones(key_corr.shape), k=1).astype(bool))
        correlations = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                val = upper_tri.loc[idx, col]
                if pd.notna(val):
                    correlations.append({
                        'var1': idx,
                        'var2': col,
                        'correlation': val
                    })
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', 
                                                         key=abs, ascending=False)
        
        # Save top correlations
        corr_df.to_csv(output_dir / f'{city_name}_top_correlations.csv', index=False)

        # Print top 25
        print(corr_df.head(25).to_string(index=False))
    
    # Create individual scatter plots for top correlations
    if len(available_key_vars) >= 2:
        print(f"\n  Creating correlation scatter plots...")
        
        # Select top 6 correlation pairs
        top_pairs = corr_df.head(6)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{city_name.upper()} - Top Correlations', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            var1, var2, corr_val = row['var1'], row['var2'], row['correlation']
            
            # Filter outliers for better visualization
            df_plot = df[[var1, var2]].copy()
            df_plot = df_plot[
                (df_plot[var1].between(df_plot[var1].quantile(0.01), 
                                       df_plot[var1].quantile(0.99))) &
                (df_plot[var2].between(df_plot[var2].quantile(0.01), 
                                       df_plot[var2].quantile(0.99)))
            ]
            
            ax.scatter(df_plot[var1], df_plot[var2], alpha=0.5, s=20)
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title(f'{var1} vs {var2}\nr = {corr_val:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_pairs), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{city_name}_correlation_scatter_plots.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved correlation scatter plots")
        plt.close()
    
    print(f"\n✅ Correlation analysis complete for {city_name}")

def analyze_city(city_folder, base_dir='.', use_detailed=False):
    """
    Complete analysis for a single city
    
    Args:
        city_folder: Name of the city folder
        base_dir: Base directory containing city folders
        use_detailed: If True, use listings.csv.gz (79 vars), else listings.csv (19 vars)
    """
    city_path = Path(base_dir) / city_folder
    city_name = city_folder
    
    # Choose which file to load based on use_detailed flag
    if use_detailed:
        # When -all is specified, ONLY try the detailed .gz file
        listings_file = city_path / 'listings.csv.gz'
        if not listings_file.exists():
            print(f"⚠️  Detailed file (listings.csv.gz) not found for {city_name}")
            print(f"    Falling back to simple listings.csv...")
            listings_file = city_path / 'listings.csv'
            if listings_file.exists():
                print(f"📊 Using SIMPLE dataset (19 variables) as fallback")
        else:
            print(f"📊 Using DETAILED dataset (79 variables) from listings.csv.gz")
    else:
        # When -all is NOT specified, use the simple file
        listings_file = city_path / 'listings.csv'
        if not listings_file.exists():
            print(f"⚠️  Simple file (listings.csv) not found for {city_name}")
            print(f"    Trying detailed listings.csv.gz...")
            listings_file = city_path / 'listings.csv.gz'
            if listings_file.exists():
                print(f"📊 Using DETAILED dataset (79 variables) as fallback")
        else:
            print(f"📊 Using SIMPLE dataset (19 variables) from listings.csv")
    
    if not listings_file.exists():
        print(f"❌ No listings file found for {city_name}")
        return None
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    try:
        if str(listings_file).endswith('.gz'):
            df = pd.read_csv(listings_file, compression='gzip')
        else:
            df = pd.read_csv(listings_file)
        print(f"✓ Loaded {len(df):,} listings with {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading {city_name}: {e}")
        return None
    
    # Clean price if present
    if 'price' in df.columns:
        df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    
    # Create output directory in city folder
    output_dir = city_path / 'analysis_output'
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create variable summary table
    print(f"\n📋 Creating variable summary table...")
    summary_table = create_variable_summary_table(df, city_name)
    summary_table.to_csv(output_dir / f'{city_name}_variable_summary.csv', index=False)
    print(f"✓ Saved: {city_name}_variable_summary.csv")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print(f"VARIABLE SUMMARY TABLE - {city_name.upper()}")
    print(f"{'='*80}")
    print(summary_table.to_string(index=False))
    
    # 2. Create correlation matrices
    create_all_correlation_matrices(df, city_name, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE FOR {city_name.upper()}")
    print(f"{'='*80}")
    print(f"Output saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - {city_name}_variable_summary.csv")
    print(f"  - {city_name}_correlation_matrix.csv")
    print(f"  - {city_name}_correlation_heatmap_full.png")
    print(f"  - {city_name}_top_correlations.csv")
    print(f"  - {city_name}_correlation_scatter_plots.png")
    
    return summary_table

def analyze_all_cities(city_folders, base_dir='.', use_detailed=False):
    """
    Run analysis for all specified cities
    
    Args:
        city_folders: List of city folder names
        base_dir: Base directory
        use_detailed: If True, use detailed 79-variable datasets
    """
    print(f"\n{'#'*80}")
    print(f"CITY-LEVEL ANALYSIS SCRIPT")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\n🔍 MODE: DETAILED ANALYSIS (79 variables from listings.csv.gz)")
    else:
        print(f"\n🔍 MODE: SIMPLE ANALYSIS (19 variables from listings.csv)")
    
    print(f"\nWill analyze {len(city_folders)} cities")
    print(f"Cities: {', '.join(city_folders)}")
    
    results = {}
    successful = 0
    failed = 0
    
    for city in city_folders:
        try:
            summary = analyze_city(city, base_dir, use_detailed=use_detailed)
            if summary is not None:
                results[city] = summary
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ ERROR analyzing {city}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"{'#'*80}")
    print(f"✅ Successfully analyzed: {successful} cities")
    print(f"❌ Failed: {failed} cities")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this script from your main Airbnb_Data directory
    
    Directory structure should be:
    Airbnb_Data/
        city_level_analysis.py  (this file)
        Austin/
            listings.csv        (19 variables - simple)
            listings.csv.gz     (79 variables - detailed)
        Boston/
            listings.csv
            listings.csv.gz
        ... etc
    
    Usage:
        python city_level_analysis.py           # Uses simple 19-column datasets
        python city_level_analysis.py -all      # Uses detailed 79-column datasets
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    
    # ====== CUSTOMIZE THIS LIST ======
    city_folders = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    # =================================
    
    # Run analysis
    results = analyze_all_cities(city_folders, base_dir='.', use_detailed=use_detailed)
    
    print(f"\n{'#'*80}")
    print(f"ALL DONE! Check each city's 'analysis_output' folder for results.")
    print(f"{'#'*80}")
    
    # Verify what was actually analyzed
    if results:
        sample_city = list(results.keys())[0]
        num_vars = len(results[sample_city])
        print(f"\n✅ VERIFICATION: Analyzed {num_vars} variables per city")
        
        if num_vars >= 70:
            print(f"   🎯 SUCCESS: Full detailed analysis with ~79 variables")
        elif num_vars <= 25:
            print(f"   ⚠️  Simple analysis with ~19 variables")
            print(f"   💡 TIP: Run 'python city_level_analysis.py -all' for full 79-variable analysis")
        else:
            print(f"   ⚠️  Partial dataset detected")
    
    if use_detailed:
        print(f"\n💡 You requested DETAILED analysis (-all flag)")
    else:
        print(f"\n💡 You used SIMPLE analysis (default)")
        print(f"   To get full analysis with 79 variables, run: python city_level_analysis.py -all")