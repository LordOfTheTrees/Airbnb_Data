"""
City-Level Analysis Script - ENHANCED WITH FEATURE ENGINEERING
Generates comprehensive statistics and correlation matrices for each city individually
NOW INCLUDES: Log transforms, within-city standardization, and revenue proxies

Run this from your main Airbnb_Data directory

⚠️ IMPORTANT: The detailed file is named listings.csv.gz (not listings_csv.gz)

Usage:
    python city_level_analysis.py                    # All cities, simple (19 vars)
    python city_level_analysis.py -all               # All cities, detailed (79 vars)
    python city_level_analysis.py Austin             # Just Austin, simple
    python city_level_analysis.py Austin -all        # Just Austin, detailed
    python city_level_analysis.py -all Chicago       # Order doesn't matter
    
Single City Mode (for debugging/development):
    python city_level_analysis.py Boston            # Fast testing on one city
    python city_level_analysis.py Boston -all       # Full analysis on one city
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

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_TEXT_LENGTH = 100  # Maximum characters for text fields in variable summary
TOP_CORRELATIONS_N = 25  # Number of top correlations to save and display
# ============================================================================

# ============================================================================
# FEATURE ENGINEERING - PRIORITY 1: LOG TRANSFORMATIONS
# ============================================================================

def add_log_transforms(df):
    """
    Priority 1: Add log transformations for key economic variables
    
    WHY: 
    - Investors think in percentages (returns), not absolute dollars
    - Price distributions are typically right-skewed
    - Log differences = percentage changes
    - Enables elasticity interpretation in correlations
    
    Transforms:
    - log_price: Natural log of price
    - log_price_per_accommodates: Natural log of price per guest capacity
    - log_reviews: Natural log of (number_of_reviews + 1) to handle zeros
    - log_accommodates: Natural log of guest capacity
    - log_beds: Natural log of (beds + 1) to handle zeros
    
    Args:
        df: DataFrame with raw listing data
    
    Returns:
        DataFrame with added log-transformed columns
    """
    print(f"\n  🔧 Priority 1: Adding log transformations...")
    
    df = df.copy()
    
    # Price transformations
    if 'price_clean' in df.columns:
        # Filter out zeros and negatives for log
        valid_price = df['price_clean'] > 0
        df.loc[valid_price, 'log_price'] = np.log(df.loc[valid_price, 'price_clean'])
        print(f"     ✓ Created log_price ({valid_price.sum():,} valid values)")
        
        # Price per accommodates (size-adjusted pricing)
        if 'accommodates' in df.columns:
            df['price_per_accommodates'] = df['price_clean'] / df['accommodates']
            valid_ppa = df['price_per_accommodates'] > 0
            df.loc[valid_ppa, 'log_price_per_accommodates'] = np.log(
                df.loc[valid_ppa, 'price_per_accommodates']
            )
            print(f"     ✓ Created log_price_per_accommodates ({valid_ppa.sum():,} valid values)")
    
    # Review transformations (use log1p to handle zeros gracefully)
    if 'number_of_reviews' in df.columns:
        df['log_reviews'] = np.log1p(df['number_of_reviews'])  # log(x + 1)
        print(f"     ✓ Created log_reviews (using log1p to handle zeros)")
    
    # Capacity transformations
    if 'accommodates' in df.columns:
        valid_acc = df['accommodates'] > 0
        df.loc[valid_acc, 'log_accommodates'] = np.log(df.loc[valid_acc, 'accommodates'])
        print(f"     ✓ Created log_accommodates ({valid_acc.sum():,} valid values)")
    
    if 'beds' in df.columns:
        df['log_beds'] = np.log1p(df['beds'])  # log(x + 1) for zeros
        print(f"     ✓ Created log_beds (using log1p to handle zeros)")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - PRIORITY 2: WITHIN-CITY STANDARDIZATION
# ============================================================================

def add_within_city_metrics(df, city_name):
    """
    Priority 2: Add within-city standardized metrics for cross-market comparison
    
    WHY:
    - A $200/night listing means different things in NYC vs Memphis
    - Investors care about relative positioning: "top 10% in this market"
    - Z-scores show how many standard deviations above/below average
    - Percentiles directly answer "better than X% of competition"
    
    Metrics:
    - price_zscore: Z-score of price within city (std devs from mean)
    - price_percentile: Percentile rank within city (0-1 scale)
    - log_price_zscore: Z-score of log price (for relative % differences)
    - reviews_zscore: Z-score of number of reviews
    - reviews_percentile: Percentile rank of reviews
    
    Args:
        df: DataFrame with listing data (should have price_clean, etc.)
        city_name: Name of the city (for display purposes)
    
    Returns:
        DataFrame with added standardized metrics
    """
    print(f"\n  🔧 Priority 2: Adding within-city standardization for {city_name}...")
    
    df = df.copy()
    
    # Price standardization
    if 'price_clean' in df.columns:
        # Z-score: how many standard deviations from mean?
        price_mean = df['price_clean'].mean()
        price_std = df['price_clean'].std()
        
        if price_std > 0:  # Avoid division by zero
            df['price_zscore'] = (df['price_clean'] - price_mean) / price_std
            print(f"     ✓ Created price_zscore (mean=${price_mean:.2f}, std=${price_std:.2f})")
        
        # Percentile rank: better than X% of listings
        df['price_percentile'] = df['price_clean'].rank(pct=True)
        print(f"     ✓ Created price_percentile (0=cheapest, 1=most expensive)")
    
    # Log price standardization (for relative percentage positioning)
    if 'log_price' in df.columns:
        log_price_mean = df['log_price'].mean()
        log_price_std = df['log_price'].std()
        
        if log_price_std > 0:
            df['log_price_zscore'] = (df['log_price'] - log_price_mean) / log_price_std
            print(f"     ✓ Created log_price_zscore (for relative % differences)")
    
    # Review count standardization (popularity/visibility metric)
    if 'number_of_reviews' in df.columns:
        reviews_mean = df['number_of_reviews'].mean()
        reviews_std = df['number_of_reviews'].std()
        
        if reviews_std > 0:
            df['reviews_zscore'] = (df['number_of_reviews'] - reviews_mean) / reviews_std
            print(f"     ✓ Created reviews_zscore (mean={reviews_mean:.1f}, std={reviews_std:.1f})")
        
        df['reviews_percentile'] = df['number_of_reviews'].rank(pct=True)
        print(f"     ✓ Created reviews_percentile (visibility ranking)")
    
    # Reviews per month standardization (activity velocity)
    if 'reviews_per_month' in df.columns:
        rpm_mean = df['reviews_per_month'].mean()
        rpm_std = df['reviews_per_month'].std()
        
        if rpm_std > 0:
            df['reviews_per_month_zscore'] = (df['reviews_per_month'] - rpm_mean) / rpm_std
            print(f"     ✓ Created reviews_per_month_zscore")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - PRIORITY 3: REVENUE PROXIES
# ============================================================================

def add_revenue_proxies(df):
    """
    Priority 3: Create revenue and ROI proxy metrics
    
    WHY:
    - Investors care about RETURNS, not just prices
    - Revenue = Price × Occupancy × Days
    - Size-adjusted metrics enable fair comparison
    - These metrics directly answer "which property makes more money?"
    
    Metrics:
    - est_annual_revenue: Estimated yearly revenue based on availability
    - revenue_per_accommodates: Revenue per guest capacity (unit economics)
    - revenue_per_bedroom: Revenue per bedroom (if bedrooms available)
    - occupancy_proxy: Estimated occupancy rate from availability
    
    NOTE: True occupancy is not available, so we use (365 - availability_365)
    as a proxy. This assumes unavailable days = booked days, which is imperfect
    but the best we can do with this data.
    
    Args:
        df: DataFrame with price_clean and availability_365
    
    Returns:
        DataFrame with added revenue proxy columns
    """
    print(f"\n  🔧 Priority 3: Adding revenue proxy metrics...")
    
    df = df.copy()
    
    # Occupancy proxy: assume unavailable days are booked days
    if 'availability_365' in df.columns:
        df['occupancy_proxy'] = (365 - df['availability_365']) / 365
        print(f"     ✓ Created occupancy_proxy (1 - availability_365/365)")
        
        # Estimated annual revenue
        if 'price_clean' in df.columns:
            df['est_annual_revenue'] = df['price_clean'] * (365 - df['availability_365'])
            print(f"     ✓ Created est_annual_revenue (price × occupied_days)")
            
            # Size-adjusted revenue metrics
            if 'accommodates' in df.columns:
                df['revenue_per_accommodates'] = df['est_annual_revenue'] / df['accommodates']
                print(f"     ✓ Created revenue_per_accommodates (unit economics)")
            
            if 'bedrooms' in df.columns:
                # Handle 0 bedrooms (studios)
                valid_br = df['bedrooms'] > 0
                df.loc[valid_br, 'revenue_per_bedroom'] = (
                    df.loc[valid_br, 'est_annual_revenue'] / df.loc[valid_br, 'bedrooms']
                )
                print(f"     ✓ Created revenue_per_bedroom ({valid_br.sum():,} valid values)")
            
            # Log-transformed revenue for percentage analysis
            valid_rev = df['est_annual_revenue'] > 0
            df.loc[valid_rev, 'log_est_revenue'] = np.log(
                df.loc[valid_rev, 'est_annual_revenue']
            )
            print(f"     ✓ Created log_est_revenue ({valid_rev.sum():,} valid values)")
    
    return df


# ============================================================================
# MASTER FEATURE ENGINEERING FUNCTION
# ============================================================================

def apply_all_feature_engineering(df, city_name):
    """
    Apply all feature engineering transformations in sequence
    
    This is the main function that orchestrates all transformations.
    Call this after loading raw data and before analysis.
    
    Args:
        df: Raw DataFrame from CSV
        city_name: Name of city (for display and within-city metrics)
    
    Returns:
        DataFrame with all engineered features added
    """
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING FOR {city_name.upper()}")
    print(f"{'='*80}")
    print(f"Starting with {len(df):,} listings and {len(df.columns)} columns")
    
    original_cols = len(df.columns)
    
    # Apply transformations in priority order
    df = add_log_transforms(df)
    df = add_within_city_metrics(df, city_name)
    df = add_revenue_proxies(df)
    
    new_cols = len(df.columns)
    added_cols = new_cols - original_cols
    
    print(f"\n  ✅ Feature engineering complete!")
    print(f"     Added {added_cols} new columns (now {new_cols} total)")
    print(f"{'='*80}")
    
    return df


# ============================================================================
# EXISTING HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

def sanitize_text(text, max_length=MAX_TEXT_LENGTH):
    """
    Sanitize text for safe CSV storage
    - Truncate to max_length
    - Remove newlines and extra spaces
    - Remove problematic characters
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

def create_all_correlation_matrices(df, city_name, output_dir, top_n=25):
    """
    Create correlation matrices for all combinations of numeric variables
    
    Args:
        df: DataFrame with city data
        city_name: Name of the city
        output_dir: Directory to save outputs
        top_n: Number of top correlations to analyze (default 25)
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
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'{city_name.upper()} - Correlation Matrix (All Variables)', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_correlation_heatmap_full.png', 
               dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved full correlation heatmap")
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
    top_corrs_df = pd.DataFrame({
        'variable_1': [pair[0] for pair in top_corrs.index],
        'variable_2': [pair[1] for pair in top_corrs.index],
        'correlation': top_corrs.values
    })
    
    # Save top correlations
    top_corrs_df.to_csv(output_dir / f'{city_name}_top_correlations.csv', index=False)
    print(f"  ✓ Saved top {len(top_corrs_df)} correlations")
    
    # Create scatter plots for top correlations
    print(f"\n  Creating scatter plots for top correlations...")
    n_plots = min(9, len(top_corrs_df))  # Up to 9 plots (3x3 grid)
    
    if n_plots > 0:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(n_plots):
            row = top_corrs_df.iloc[idx]
            var1, var2, corr = row['variable_1'], row['variable_2'], row['correlation']
            
            # Create scatter plot
            axes[idx].scatter(df[var1], df[var2], alpha=0.3, s=20)
            axes[idx].set_xlabel(var1, fontsize=9)
            axes[idx].set_ylabel(var2, fontsize=9)
            axes[idx].set_title(f'r = {corr:.3f}', fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, 9):
            axes[idx].axis('off')
        
        plt.suptitle(f'{city_name.upper()} - Top Correlation Scatter Plots', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / f'{city_name}_correlation_scatter_plots.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved scatter plots")
        plt.close()


# ============================================================================
# MAIN CITY ANALYSIS FUNCTION (MODIFIED TO INCLUDE FEATURE ENGINEERING)
# ============================================================================

def analyze_city(city_name, base_dir='.', use_detailed=False):
    """
    Analyze a single city with feature engineering
    
    Modified to include Priority 1-3 feature engineering before analysis
    """
    city_path = Path(base_dir) / city_name
    
    if not city_path.exists():
        print(f"⚠️  Directory not found: {city_path}")
        return None
    
    # Determine which file to use
    if use_detailed:
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
    
    # *** NEW: APPLY FEATURE ENGINEERING ***
    df = apply_all_feature_engineering(df, city_name)
    
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
    create_all_correlation_matrices(df, city_name, output_dir, top_n=TOP_CORRELATIONS_N)
    
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
    print(f"CITY-LEVEL ANALYSIS SCRIPT - WITH FEATURE ENGINEERING")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\n🔍 MODE: DETAILED ANALYSIS (79 variables from listings_csv.gz)")
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
            listings.csv.gz     (79 variables - detailed)  ⚠️ NOTE: .csv.gz not _csv.gz
        Boston/
            listings.csv
            listings.csv.gz
        ... etc
    
    Usage:
        python city_level_analysis.py                    # All cities, simple datasets
        python city_level_analysis.py -all               # All cities, detailed datasets
        python city_level_analysis.py Austin             # Just Austin, simple dataset
        python city_level_analysis.py Austin -all        # Just Austin, detailed dataset
        python city_level_analysis.py -all Austin        # Order doesn't matter
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    
    # Check if a specific city was requested
    city_args = [arg for arg in sys.argv[1:] if arg != '-all']
    single_city = city_args[0] if city_args else None
    
    # ====== CUSTOMIZE THIS LIST ======
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    # =================================
    
    # Determine which cities to analyze
    if single_city:
        # Single city mode (case-insensitive matching)
        city_folders = [c for c in all_cities if c.lower() == single_city.lower()]
        
        if not city_folders:
            print(f"\n❌ ERROR: City '{single_city}' not found in city list!")
            print(f"\nAvailable cities:")
            for city in all_cities:
                print(f"  - {city}")
            print(f"\nUsage: python city_level_analysis.py CityName [-all]")
            sys.exit(1)
        
        print(f"\n🎯 SINGLE CITY MODE: Analyzing {city_folders[0]} only")
        print(f"   (For all cities, run without city name)")
    else:
        # All cities mode
        city_folders = all_cities
        print(f"\n📊 BATCH MODE: Analyzing all {len(city_folders)} cities")
    
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
            print(f"   🎯 SUCCESS: Full detailed analysis with ~79+ variables")
        elif num_vars <= 25:
            print(f"   ⚠️  Simple analysis with ~19+ variables")
            print(f"   💡 TIP: Run 'python city_level_analysis.py -all' for full 79-variable analysis")
        else:
            print(f"   ⚠️  Partial dataset detected")
    
    if use_detailed:
        print(f"\n💡 You requested DETAILED analysis (-all flag)")
    else:
        print(f"\n💡 You used SIMPLE analysis (default)")
        print(f"   To get full analysis with 79 variables, run: python city_level_analysis.py -all")