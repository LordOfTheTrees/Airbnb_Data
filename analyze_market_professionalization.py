"""
Market Professionalization Analysis
Ranks markets by professionalization level based on host operator density

Usage:
    python analyze_market_professionalization.py -all  # All cities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import io

# Import feature engineering functions
from city_level_analysis import apply_all_feature_engineering

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_city_data(city_name, base_dir='.', use_detailed=False):
    """Load and prepare data for a single city"""
    city_path = Path(base_dir) / city_name
    
    if not city_path.exists():
        raise FileNotFoundError(f"City directory not found: {city_path}")
    
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
        raise FileNotFoundError(f"No listings file found for {city_name}")
    
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
    df = apply_all_feature_engineering(df, city_name)
    
    return df


def calculate_market_professionalization_metrics(df, city_name):
    """
    Calculate comprehensive professionalization metrics for a market
    
    Returns dict with all metrics
    """
    metrics = {'city': city_name}
    
    if 'calculated_host_listings_count' not in df.columns:
        return None
    
    # Convert to numeric if needed
    if df['calculated_host_listings_count'].dtype == 'object':
        df['calculated_host_listings_count'] = pd.to_numeric(
            df['calculated_host_listings_count'], errors='coerce'
        )
    
    total_listings = len(df)
    metrics['total_listings'] = total_listings
    
    # Professional host metrics
    professional_listings = (df['calculated_host_listings_count'] >= 2).sum()
    large_operator_listings = (df['calculated_host_listings_count'] >= 21).sum()
    mega_operator_listings = (df['calculated_host_listings_count'] >= 50).sum()
    
    metrics['pct_professional'] = (professional_listings / total_listings) * 100
    metrics['pct_large_operators'] = (large_operator_listings / total_listings) * 100
    metrics['pct_mega_operators'] = (mega_operator_listings / total_listings) * 100
    
    # Host distribution metrics
    metrics['median_host_listings'] = df['calculated_host_listings_count'].median()
    metrics['mean_host_listings'] = df['calculated_host_listings_count'].mean()
    metrics['p75_host_listings'] = df['calculated_host_listings_count'].quantile(0.75)
    metrics['p95_host_listings'] = df['calculated_host_listings_count'].quantile(0.95)
    metrics['max_host_listings'] = df['calculated_host_listings_count'].max()
    
    # Unique hosts
    unique_hosts = df['host_id'].nunique()
    metrics['unique_hosts'] = unique_hosts
    metrics['listings_per_host'] = total_listings / unique_hosts if unique_hosts > 0 else 0
    
    # Gini coefficient for concentration
    host_listing_counts = df['calculated_host_listings_count'].dropna()
    if len(host_listing_counts) > 1:
        sorted_counts = np.sort(host_listing_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        total = cumsum[-1]
        
        if total > 0:
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * total) - (n + 1) / n
            gini = abs(gini)
            gini = min(gini, 1.0)
        else:
            gini = 0
    else:
        gini = 0
    
    metrics['gini_coefficient'] = gini
    
    # Herfindahl-Hirschman Index (HHI) - another concentration measure
    # HHI = sum of squared market shares (0-1 scale, higher = more concentrated)
    host_counts = df['calculated_host_listings_count'].value_counts()
    if len(host_counts) > 0 and total_listings > 0:
        market_shares = host_counts / total_listings
        hhi = (market_shares ** 2).sum()
    else:
        hhi = 0
    metrics['hhi_index'] = hhi
    
    # Professionalization score (composite)
    pct_prof = metrics['pct_professional']
    pct_large = metrics['pct_large_operators']
    median_listings = metrics['median_host_listings']
    
    # Normalize components
    pct_large_scaled = min(pct_large * 2, 100)  # Scale assuming max ~50%
    median_scaled = min(median_listings / 50 * 100, 100)  # Scale assuming max ~50
    gini_scaled = gini * 100
    
    # Weighted composite score
    professionalization_score = (
        0.3 * pct_prof +
        0.3 * pct_large_scaled +
        0.2 * median_scaled +
        0.2 * gini_scaled
    )
    metrics['professionalization_score'] = professionalization_score
    
    # Additional insights
    # % of market controlled by top 10% of hosts
    if len(host_listing_counts) > 0:
        top_10_pct_threshold = host_listing_counts.quantile(0.9)
        top_10_pct_listings = df[df['calculated_host_listings_count'] >= top_10_pct_threshold]['calculated_host_listings_count'].sum()
        metrics['pct_market_top_10pct_hosts'] = (top_10_pct_listings / total_listings) * 100 if total_listings > 0 else 0
    else:
        metrics['pct_market_top_10pct_hosts'] = 0
    
    return metrics


def rank_markets_by_professionalization(all_cities, base_dir='.', use_detailed=False):
    """
    Analyze and rank all markets by professionalization level
    
    Args:
        all_cities: List of city names
        base_dir: Base directory
        use_detailed: Use detailed dataset
    
    Returns:
        DataFrame with ranked markets
    """
    print(f"\n{'#'*80}")
    print(f"MARKET PROFESSIONALIZATION ANALYSIS")
    print(f"{'#'*80}")
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for city in all_cities:
        print(f"\n{'='*80}")
        print(f"Processing: {city.upper()}")
        print(f"{'='*80}")
        
        try:
            df = load_city_data(city, base_dir=base_dir, use_detailed=use_detailed)
            metrics = calculate_market_professionalization_metrics(df, city)
            
            if metrics:
                all_metrics.append(metrics)
                successful += 1
                print(f"✓ {city}: {metrics['pct_professional']:.1f}% professional, "
                     f"Score: {metrics['professionalization_score']:.1f}/100")
            else:
                print(f"⚠️  {city}: Missing required data")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR processing {city}: {e}")
            failed += 1
    
    if not all_metrics:
        print("\nNo metrics collected. Check data availability.")
        return None
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Sort by professionalization score (descending)
    metrics_df = metrics_df.sort_values('professionalization_score', ascending=False)
    
    print(f"\n{'#'*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'#'*80}")
    print(f"Successfully analyzed: {successful} cities")
    print(f"Failed: {failed} cities")
    
    return metrics_df


def create_professionalization_visualizations(metrics_df, output_dir='city_comparison_outputs'):
    """Create visualizations of market professionalization"""
    print(f"\n{'='*80}")
    print(f"CREATING PROFESSIONALIZATION VISUALIZATIONS")
    print(f"{'='*80}")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Professionalization Score Ranking
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Professionalization Score by City
    ax = axes[0, 0]
    metrics_sorted = metrics_df.sort_values('professionalization_score', ascending=True)
    ax.barh(range(len(metrics_sorted)), metrics_sorted['professionalization_score'])
    ax.set_yticks(range(len(metrics_sorted)))
    ax.set_yticklabels(metrics_sorted['city'], fontsize=10)
    ax.set_xlabel('Professionalization Score (0-100)', fontweight='bold', fontsize=12)
    ax.set_title('Market Professionalization Score by City', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: % Professional Hosts
    ax = axes[0, 1]
    metrics_sorted2 = metrics_df.sort_values('pct_professional', ascending=True)
    ax.barh(range(len(metrics_sorted2)), metrics_sorted2['pct_professional'], color='steelblue')
    ax.set_yticks(range(len(metrics_sorted2)))
    ax.set_yticklabels(metrics_sorted2['city'], fontsize=10)
    ax.set_xlabel('% Professional Hosts (2+ listings)', fontweight='bold', fontsize=12)
    ax.set_title('Percentage of Professional Hosts by City', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: % Large Operators
    ax = axes[1, 0]
    metrics_sorted3 = metrics_df.sort_values('pct_large_operators', ascending=True)
    ax.barh(range(len(metrics_sorted3)), metrics_sorted3['pct_large_operators'], color='coral')
    ax.set_yticks(range(len(metrics_sorted3)))
    ax.set_yticklabels(metrics_sorted3['city'], fontsize=10)
    ax.set_xlabel('% Large Operators (21+ listings)', fontweight='bold', fontsize=12)
    ax.set_title('Percentage of Large Operators by City', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Gini Coefficient (concentration)
    ax = axes[1, 1]
    metrics_sorted4 = metrics_df.sort_values('gini_coefficient', ascending=True)
    ax.barh(range(len(metrics_sorted4)), metrics_sorted4['gini_coefficient'], color='green')
    ax.set_yticks(range(len(metrics_sorted4)))
    ax.set_yticklabels(metrics_sorted4['city'], fontsize=10)
    ax.set_xlabel('Gini Coefficient (0=even, 1=concentrated)', fontweight='bold', fontsize=12)
    ax.set_title('Market Concentration (Gini Coefficient)', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Market Professionalization Analysis - All Cities', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_professionalization_analysis.png', 
               dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/market_professionalization_analysis.png")
    plt.close()
    
    # Figure 2: Scatter plots showing relationships
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Professionalization Score vs % Professional
    ax = axes[0, 0]
    ax.scatter(metrics_df['pct_professional'], metrics_df['professionalization_score'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1)
    for _, row in metrics_df.iterrows():
        ax.annotate(row['city'], (row['pct_professional'], row['professionalization_score']),
                   fontsize=8, alpha=0.7)
    ax.set_xlabel('% Professional Hosts', fontweight='bold', fontsize=12)
    ax.set_ylabel('Professionalization Score', fontweight='bold', fontsize=12)
    ax.set_title('Professionalization Score vs % Professional Hosts', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    corr = metrics_df['pct_professional'].corr(metrics_df['professionalization_score'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Gini vs % Large Operators
    ax = axes[0, 1]
    ax.scatter(metrics_df['pct_large_operators'], metrics_df['gini_coefficient'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1, color='coral')
    for _, row in metrics_df.iterrows():
        ax.annotate(row['city'], (row['pct_large_operators'], row['gini_coefficient']),
                   fontsize=8, alpha=0.7)
    ax.set_xlabel('% Large Operators (21+)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Gini Coefficient', fontweight='bold', fontsize=12)
    ax.set_title('Market Concentration vs Large Operators', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    corr = metrics_df['pct_large_operators'].corr(metrics_df['gini_coefficient'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Median Listings vs Market Size
    ax = axes[1, 0]
    ax.scatter(metrics_df['total_listings'], metrics_df['median_host_listings'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1, color='green')
    for _, row in metrics_df.iterrows():
        ax.annotate(row['city'], (row['total_listings'], row['median_host_listings']),
                   fontsize=8, alpha=0.7)
    ax.set_xlabel('Total Listings (Market Size)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Median Listings per Host', fontweight='bold', fontsize=12)
    ax.set_title('Median Host Listings vs Market Size', fontweight='bold', fontsize=13)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    corr = np.log(metrics_df['total_listings']).corr(metrics_df['median_host_listings'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: HHI vs Gini (concentration measures)
    ax = axes[1, 1]
    ax.scatter(metrics_df['gini_coefficient'], metrics_df['hhi_index'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1, color='purple')
    for _, row in metrics_df.iterrows():
        ax.annotate(row['city'], (row['gini_coefficient'], row['hhi_index']),
                   fontsize=8, alpha=0.7)
    ax.set_xlabel('Gini Coefficient', fontweight='bold', fontsize=12)
    ax.set_ylabel('HHI Index', fontweight='bold', fontsize=12)
    ax.set_title('Concentration Measures: Gini vs HHI', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    corr = metrics_df['gini_coefficient'].corr(metrics_df['hhi_index'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Market Professionalization Relationships', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_professionalization_relationships.png', 
               dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/market_professionalization_relationships.png")
    plt.close()


def main():
    """Main execution function"""
    use_detailed = '-all' in sys.argv
    
    # List of all cities
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    
    print(f"\n{'#'*80}")
    print(f"MARKET PROFESSIONALIZATION ANALYSIS")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\nMODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"\nMODE: SIMPLE ANALYSIS (19 variables)")
        print("WARNING: Professionalization metrics require detailed dataset")
        print("Run with -all flag for full analysis")
    
    # Rank markets
    metrics_df = rank_markets_by_professionalization(all_cities, base_dir='.', use_detailed=use_detailed)
    
    if metrics_df is not None and len(metrics_df) > 0:
        # Save results
        output_dir = Path('city_comparison_outputs')
        output_dir.mkdir(exist_ok=True)
        
        metrics_df.to_csv(output_dir / 'market_professionalization_ranking.csv', index=False)
        print(f"\n✓ Saved: {output_dir}/market_professionalization_ranking.csv")
        
        # Create visualizations
        create_professionalization_visualizations(metrics_df, output_dir=str(output_dir))
        
        # Print ranking table
        print(f"\n{'='*80}")
        print(f"MARKET PROFESSIONALIZATION RANKING")
        print(f"{'='*80}")
        print("\nTop 10 Most Professional Markets:")
        display_cols = ['city', 'professionalization_score', 'pct_professional', 
                       'pct_large_operators', 'median_host_listings', 'gini_coefficient']
        print(metrics_df[display_cols].head(10).to_string(index=False))
        
        print(f"\n{'='*80}")
        print(f"KEY INSIGHTS")
        print(f"{'='*80}")
        print(f"Most Professional: {metrics_df.iloc[0]['city']} (Score: {metrics_df.iloc[0]['professionalization_score']:.1f})")
        print(f"Least Professional: {metrics_df.iloc[-1]['city']} (Score: {metrics_df.iloc[-1]['professionalization_score']:.1f})")
        print(f"\nMean Professionalization Score: {metrics_df['professionalization_score'].mean():.1f}")
        print(f"Median: {metrics_df['professionalization_score'].median():.1f}")
        print(f"Std Dev: {metrics_df['professionalization_score'].std():.1f}")


if __name__ == "__main__":
    main()

