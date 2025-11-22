"""
Analyze if the spike at 255 matches the missing mass above 0.7
Check if capped listings' calendar proxy values match expected distribution
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


def analyze_cap_mass_balance(df, city_name):
    """Check if mass at cap equals missing mass above 0.7"""
    
    # Convert to numeric
    if 'estimated_occupancy_l365d' in df.columns:
        if df['estimated_occupancy_l365d'].dtype == 'object':
            df['estimated_occupancy_l365d'] = pd.to_numeric(
                df['estimated_occupancy_l365d'], errors='coerce'
            )
    
    # Calculate metrics
    df['occupancy_rate'] = df['estimated_occupancy_l365d'] / 365
    if 'availability_365' in df.columns:
        df['calendar_occupancy_proxy'] = (365 - df['availability_365']) / 365
    
    # Filter to valid data
    valid = df[['estimated_occupancy_l365d', 'occupancy_rate', 'calendar_occupancy_proxy']].dropna()
    
    if len(valid) == 0:
        return None
    
    # Identify capped listings
    capped = valid[valid['estimated_occupancy_l365d'] == 255]
    uncapped = valid[valid['estimated_occupancy_l365d'] < 255]
    
    # Key metrics
    total_valid = len(valid)
    capped_count = len(capped)
    capped_pct = capped_count / total_valid * 100 if total_valid > 0 else 0
    
    # Calendar proxy > 0.7
    total_cal_above_07 = (valid['calendar_occupancy_proxy'] > 0.7).sum()
    total_actual_above_07 = (valid['occupancy_rate'] > 0.7).sum()
    missing_mass = total_cal_above_07 - total_actual_above_07
    
    # Ratio: missing mass / capped listings
    ratio = missing_mass / capped_count if capped_count > 0 else 0
    
    # For capped listings, what are their calendar proxy values?
    if len(capped) > 0:
        capped_cal_above_07 = (capped['calendar_occupancy_proxy'] > 0.7).sum()
        capped_cal_mean = capped['calendar_occupancy_proxy'].mean()
        capped_cal_median = capped['calendar_occupancy_proxy'].median()
    else:
        capped_cal_above_07 = 0
        capped_cal_mean = np.nan
        capped_cal_median = np.nan
    
    # For uncapped listings with calendar proxy > 0.7
    if len(uncapped) > 0:
        uncapped_cal_above_07 = (uncapped['calendar_occupancy_proxy'] > 0.7).sum()
        uncapped_actual_above_07 = (uncapped['occupancy_rate'] > 0.7).sum()
    else:
        uncapped_cal_above_07 = 0
        uncapped_actual_above_07 = 0
    
    # Expected distribution: listings with calendar proxy > 0.7
    expected_above_07 = valid[valid['calendar_occupancy_proxy'] > 0.7]
    if len(expected_above_07) > 0:
        expected_capped = (expected_above_07['estimated_occupancy_l365d'] == 255).sum()
        expected_uncapped = (expected_above_07['estimated_occupancy_l365d'] < 255).sum()
        expected_capped_pct = expected_capped / len(expected_above_07) * 100
    else:
        expected_capped = 0
        expected_uncapped = 0
        expected_capped_pct = 0
    
    return {
        'city': city_name,
        'total_valid': total_valid,
        'capped_count': capped_count,
        'capped_pct': capped_pct,
        'total_cal_above_07': total_cal_above_07,
        'total_cal_above_07_pct': total_cal_above_07 / total_valid * 100 if total_valid > 0 else 0,
        'total_actual_above_07': total_actual_above_07,
        'missing_mass': missing_mass,
        'missing_mass_pct': missing_mass / total_valid * 100 if total_valid > 0 else 0,
        'ratio': ratio,
        'capped_cal_above_07': capped_cal_above_07,
        'capped_cal_above_07_pct': capped_cal_above_07 / capped_count * 100 if capped_count > 0 else 0,
        'capped_cal_mean': capped_cal_mean,
        'capped_cal_median': capped_cal_median,
        'uncapped_cal_above_07': uncapped_cal_above_07,
        'uncapped_actual_above_07': uncapped_actual_above_07,
        'expected_capped': expected_capped,
        'expected_uncapped': expected_uncapped,
        'expected_capped_pct': expected_capped_pct
    }


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
    print(f"CAP MASS BALANCE ANALYSIS")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\nMODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"\nMODE: SIMPLE ANALYSIS (19 variables)")
        print("WARNING: Need detailed dataset for full analysis")
    
    all_results = []
    
    for city in all_cities:
        print(f"\n{'='*80}")
        print(f"Processing: {city.upper()}")
        print(f"{'='*80}")
        
        try:
            df = load_city_data(city, base_dir='.', use_detailed=use_detailed)
            result = analyze_cap_mass_balance(df, city)
            
            if result:
                all_results.append(result)
                print(f"✓ {city}:")
                print(f"  Capped: {result['capped_count']:,} ({result['capped_pct']:.1f}%)")
                print(f"  Calendar > 0.7: {result['total_cal_above_07']:,} ({result['total_cal_above_07_pct']:.1f}%)")
                print(f"  Missing mass: {result['missing_mass']:,} ({result['missing_mass_pct']:.1f}%)")
                print(f"  Ratio (missing/capped): {result['ratio']:.2f}")
                print(f"  Capped with cal > 0.7: {result['capped_cal_above_07']:,} ({result['capped_cal_above_07_pct']:.1f}%)")
                print(f"  Expected capped %: {result['expected_capped_pct']:.1f}%")
            else:
                print(f"⚠️  {city}: No valid data")
                
        except Exception as e:
            print(f"❌ ERROR processing {city}: {e}")
    
    if not all_results:
        print("\nNo results collected. Check data availability.")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_dir = Path('city_comparison_outputs')
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / 'cap_mass_balance_analysis.csv', index=False)
    print(f"\n✓ Saved: {output_dir}/cap_mass_balance_analysis.csv")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nRatio (Missing Mass / Capped Listings):")
    print(f"  Mean: {results_df['ratio'].mean():.2f}")
    print(f"  Median: {results_df['ratio'].median():.2f}")
    print(f"  Min: {results_df['ratio'].min():.2f}")
    print(f"  Max: {results_df['ratio'].max():.2f}")
    print(f"  Std: {results_df['ratio'].std():.2f}")
    
    print(f"\nInterpretation:")
    print(f"  Ratio ≈ 1.0: Capped listings roughly match missing mass")
    print(f"  Ratio > 1.0: More missing mass than capped (some uncapped also > 0.7)")
    print(f"  Ratio < 1.0: Fewer missing mass than capped (some capped < 0.7)")
    
    print(f"\n{'='*80}")
    print(f"CAPPED LISTINGS - CALENDAR PROXY ANALYSIS")
    print(f"{'='*80}")
    print(f"Mean calendar proxy for capped listings: {results_df['capped_cal_mean'].mean():.3f}")
    print(f"Median calendar proxy for capped listings: {results_df['capped_cal_median'].median():.3f}")
    print(f"Mean % of capped listings with calendar > 0.7: {results_df['capped_cal_above_07_pct'].mean():.1f}%")
    
    print(f"\n{'='*80}")
    print(f"EXPECTED vs ACTUAL")
    print(f"{'='*80}")
    print(f"Of listings with calendar proxy > 0.7:")
    print(f"  Mean % that are capped: {results_df['expected_capped_pct'].mean():.1f}%")
    print(f"  Mean % that are uncapped: {100 - results_df['expected_capped_pct'].mean():.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Ratio distribution
    ax = axes[0, 0]
    ax.barh(range(len(results_df)), results_df['ratio'].sort_values(ascending=True))
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df.sort_values('ratio')['city'], fontsize=9)
    ax.set_xlabel('Ratio (Missing Mass / Capped Listings)', fontweight='bold', fontsize=11)
    ax.set_title('Mass Balance Ratio by City', fontweight='bold', fontsize=12)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect Match (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Capped % vs Missing Mass %
    ax = axes[0, 1]
    ax.scatter(results_df['capped_pct'], results_df['missing_mass_pct'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1)
    for _, row in results_df.iterrows():
        ax.annotate(row['city'], (row['capped_pct'], row['missing_mass_pct']),
                   fontsize=8, alpha=0.7)
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1, alpha=0.5, label='Perfect Match')
    ax.set_xlabel('% Capped at 255', fontweight='bold', fontsize=11)
    ax.set_ylabel('% Missing Mass (Calendar > 0.7)', fontweight='bold', fontsize=11)
    ax.set_title('Capped % vs Missing Mass %', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Calendar proxy distribution for capped listings
    ax = axes[1, 0]
    ax.barh(range(len(results_df)), results_df['capped_cal_mean'].sort_values(ascending=True))
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df.sort_values('capped_cal_mean')['city'], fontsize=9)
    ax.set_xlabel('Mean Calendar Proxy for Capped Listings', fontweight='bold', fontsize=11)
    ax.set_title('Calendar Proxy Distribution (Capped Listings)', fontweight='bold', fontsize=12)
    ax.axvline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.7 threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Expected capped % vs Actual capped %
    ax = axes[1, 1]
    # Calculate actual capped % of listings with calendar > 0.7
    results_df['actual_capped_pct'] = (results_df['capped_cal_above_07'] / 
                                       results_df['total_cal_above_07'] * 100).fillna(0)
    ax.scatter(results_df['expected_capped_pct'], results_df['actual_capped_pct'], 
              s=100, alpha=0.6, edgecolors='black', linewidth=1)
    for _, row in results_df.iterrows():
        ax.annotate(row['city'], (row['expected_capped_pct'], row['actual_capped_pct']),
                   fontsize=8, alpha=0.7)
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1, alpha=0.5, label='Perfect Match')
    ax.set_xlabel('Expected Capped % (of calendar > 0.7)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual Capped % (of calendar > 0.7)', fontweight='bold', fontsize=11)
    ax.set_title('Expected vs Actual Capping Pattern', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Cap Mass Balance Analysis - All Cities', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'cap_mass_balance_analysis.png', 
               dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/cap_mass_balance_analysis.png")
    plt.close()
    
    # Print detailed results table
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS TABLE")
    print(f"{'='*80}")
    display_cols = ['city', 'capped_pct', 'missing_mass_pct', 'ratio', 
                    'capped_cal_mean', 'expected_capped_pct']
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

