"""
Professionalization Correlation Analysis
Analyzes how professionalization correlates with occupancy and pricing

Also investigates occupancy rate distribution issues

Usage:
    python analyze_professionalization_correlations.py -all  # All cities
    python analyze_professionalization_correlations.py Austin  # Single city
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


def investigate_occupancy_distribution(df, city_name, output_dir=None):
    """
    Investigate the occupancy rate distribution issue
    Check if estimated_occupancy_l365d is capped or has data quality issues
    """
    print(f"\n{'='*80}")
    print(f"INVESTIGATING OCCUPANCY DISTRIBUTION FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    if output_dir is None:
        output_dir = Path('.') / city_name / 'analysis_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_analysis = df.copy()
    
    # Check raw estimated_occupancy_l365d values
    if 'estimated_occupancy_l365d' not in df_analysis.columns:
        print("WARNING: estimated_occupancy_l365d not found")
        return None
    
    # Convert to numeric if needed
    if df_analysis['estimated_occupancy_l365d'].dtype == 'object':
        df_analysis['estimated_occupancy_l365d'] = pd.to_numeric(
            df_analysis['estimated_occupancy_l365d'], errors='coerce'
        )
    
    # Calculate occupancy_rate
    df_analysis['occupancy_rate'] = df_analysis['estimated_occupancy_l365d'] / 365
    
    # Calendar proxy for comparison
    if 'availability_365' in df_analysis.columns:
        df_analysis['calendar_occupancy_proxy'] = (365 - df_analysis['availability_365']) / 365
    
    # Filter to valid data
    valid_data = df_analysis[['estimated_occupancy_l365d', 'occupancy_rate']].dropna()
    
    if len(valid_data) == 0:
        print("No valid occupancy data found")
        return None
    
    print(f"\nUsing {len(valid_data):,} listings with occupancy data")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"RAW estimated_occupancy_l365d STATISTICS")
    print(f"{'='*80}")
    print(f"Min: {valid_data['estimated_occupancy_l365d'].min():.0f} days")
    print(f"Max: {valid_data['estimated_occupancy_l365d'].max():.0f} days")
    print(f"Mean: {valid_data['estimated_occupancy_l365d'].mean():.1f} days")
    print(f"Median: {valid_data['estimated_occupancy_l365d'].median():.1f} days")
    print(f"Std: {valid_data['estimated_occupancy_l365d'].std():.1f} days")
    print(f"95th percentile: {valid_data['estimated_occupancy_l365d'].quantile(0.95):.0f} days")
    print(f"99th percentile: {valid_data['estimated_occupancy_l365d'].quantile(0.99):.0f} days")
    
    # Check for capping
    max_days = valid_data['estimated_occupancy_l365d'].max()
    at_max = (valid_data['estimated_occupancy_l365d'] == max_days).sum()
    print(f"\nListings at maximum value ({max_days:.0f} days): {at_max:,} ({at_max/len(valid_data)*100:.1f}%)")
    
    # Check for values near 255 days (common cap)
    near_255 = ((valid_data['estimated_occupancy_l365d'] >= 250) & 
                (valid_data['estimated_occupancy_l365d'] <= 260)).sum()
    if near_255 > 0:
        print(f"Listings near 255 days (250-260): {near_255:,} ({near_255/len(valid_data)*100:.1f}%)")
    
    # Occupancy rate statistics
    print(f"\n{'='*80}")
    print(f"OCCUPANCY RATE STATISTICS (estimated_occupancy_l365d / 365)")
    print(f"{'='*80}")
    print(f"Min: {valid_data['occupancy_rate'].min():.3f} ({valid_data['occupancy_rate'].min()*100:.1f}%)")
    print(f"Max: {valid_data['occupancy_rate'].max():.3f} ({valid_data['occupancy_rate'].max()*100:.1f}%)")
    print(f"Mean: {valid_data['occupancy_rate'].mean():.3f} ({valid_data['occupancy_rate'].mean()*100:.1f}%)")
    print(f"Median: {valid_data['occupancy_rate'].median():.3f} ({valid_data['occupancy_rate'].median()*100:.1f}%)")
    print(f"Std: {valid_data['occupancy_rate'].std():.3f}")
    
    # Check distribution around 0.7
    near_07 = ((valid_data['occupancy_rate'] >= 0.68) & 
               (valid_data['occupancy_rate'] <= 0.72)).sum()
    print(f"\nListings with occupancy rate 0.68-0.72: {near_07:,} ({near_07/len(valid_data)*100:.1f}%)")
    
    above_07 = (valid_data['occupancy_rate'] > 0.7).sum()
    print(f"Listings with occupancy rate > 0.7: {above_07:,} ({above_07/len(valid_data)*100:.1f}%)")
    
    above_08 = (valid_data['occupancy_rate'] > 0.8).sum()
    print(f"Listings with occupancy rate > 0.8: {above_08:,} ({above_08/len(valid_data)*100:.1f}%)")
    
    # Compare with calendar proxy if available
    if 'calendar_occupancy_proxy' in df_analysis.columns:
        cal_valid = df_analysis[['calendar_occupancy_proxy', 'occupancy_rate']].dropna()
        if len(cal_valid) > 0:
            print(f"\n{'='*80}")
            print(f"CALENDAR PROXY COMPARISON")
            print(f"{'='*80}")
            print(f"Calendar proxy max: {cal_valid['calendar_occupancy_proxy'].max():.3f} ({cal_valid['calendar_occupancy_proxy'].max()*100:.1f}%)")
            print(f"Actual occupancy max: {cal_valid['occupancy_rate'].max():.3f} ({cal_valid['occupancy_rate'].max()*100:.1f}%)")
            print(f"Difference: {(cal_valid['calendar_occupancy_proxy'].max() - cal_valid['occupancy_rate'].max())*100:.1f} percentage points")
            
            cal_above_07 = (cal_valid['calendar_occupancy_proxy'] > 0.7).sum()
            print(f"\nCalendar proxy > 0.7: {cal_above_07:,} ({cal_above_07/len(cal_valid)*100:.1f}%)")
            print(f"Actual occupancy > 0.7: {above_07:,} ({above_07/len(cal_valid)*100:.1f}%)")
    
    # Create detailed distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Histogram of raw estimated_occupancy_l365d
    ax = axes[0, 0]
    ax.hist(valid_data['estimated_occupancy_l365d'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(255, color='red', linestyle='--', linewidth=2, label='255 days (potential cap)')
    ax.set_xlabel('Estimated Occupancy (days)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Distribution of estimated_occupancy_l365d (Raw Days)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of occupancy_rate
    ax = axes[0, 1]
    ax.hist(valid_data['occupancy_rate'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='0.7 (70%)')
    ax.set_xlabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Distribution of Occupancy Rate', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Zoomed histogram around 0.7
    ax = axes[0, 2]
    zoom_data = valid_data[(valid_data['occupancy_rate'] >= 0.5) & (valid_data['occupancy_rate'] <= 1.0)]
    if len(zoom_data) > 0:
        ax.hist(zoom_data['occupancy_rate'], bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='0.7 (70%)')
        ax.set_xlabel('Occupancy Rate', fontweight='bold', fontsize=11)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax.set_title('Occupancy Rate Distribution (0.5-1.0 zoom)', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: CDF comparison
    ax = axes[1, 0]
    sorted_occ = np.sort(valid_data['occupancy_rate'].values)
    y_vals = np.arange(1, len(sorted_occ) + 1) / len(sorted_occ)
    ax.plot(sorted_occ, y_vals, linewidth=2, label='Occupancy Rate (actual)')
    
    if 'calendar_occupancy_proxy' in df_analysis.columns:
        cal_valid = df_analysis[['calendar_occupancy_proxy']].dropna()
        if len(cal_valid) > 0:
            sorted_cal = np.sort(cal_valid['calendar_occupancy_proxy'].values)
            y_cal = np.arange(1, len(sorted_cal) + 1) / len(sorted_cal)
            ax.plot(sorted_cal, y_cal, linewidth=2, label='Calendar Proxy', linestyle='--')
    
    ax.axvline(0.7, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=11)
    ax.set_title('Cumulative Distribution Function', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Box plot comparison
    ax = axes[1, 1]
    plot_data = []
    labels = []
    
    plot_data.append(valid_data['occupancy_rate'].values)
    labels.append('Occupancy Rate\n(actual)')
    
    if 'calendar_occupancy_proxy' in df_analysis.columns:
        cal_valid = df_analysis[['calendar_occupancy_proxy']].dropna()
        if len(cal_valid) > 0:
            plot_data.append(cal_valid['calendar_occupancy_proxy'].values)
            labels.append('Calendar Proxy')
    
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_title('Box Plot Comparison', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Scatter of raw days vs rate
    ax = axes[1, 2]
    sample_size = min(5000, len(valid_data))
    sample_data = valid_data.sample(n=sample_size, random_state=42)
    ax.scatter(sample_data['estimated_occupancy_l365d'], sample_data['occupancy_rate'], 
              alpha=0.3, s=10, edgecolors='none')
    ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(255, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Estimated Occupancy (days)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_title('Raw Days vs Occupancy Rate', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Occupancy Distribution Investigation: {city_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_occupancy_distribution_investigation.png', 
               dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/{city_name}_occupancy_distribution_investigation.png")
    plt.close()
    
    return {
        'max_days': max_days,
        'max_rate': valid_data['occupancy_rate'].max(),
        'pct_above_07': above_07 / len(valid_data) * 100,
        'pct_above_08': above_08 / len(valid_data) * 100,
        'at_max_days': at_max,
        'pct_at_max': at_max / len(valid_data) * 100
    }


def analyze_professionalization_correlations(df, city_name, output_dir=None):
    """
    Analyze correlations between professionalization metrics and:
    - Occupancy rates
    - Pricing
    - Revenue
    """
    print(f"\n{'='*80}")
    print(f"PROFESSIONALIZATION CORRELATION ANALYSIS FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    if output_dir is None:
        output_dir = Path('.') / city_name / 'analysis_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_analysis = df.copy()
    
    # Required columns
    required_cols = ['host_is_professional', 'host_listings_in_city', 
                    'occupancy_rate', 'price_clean', 'log_price']
    
    missing_cols = [col for col in required_cols if col not in df_analysis.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        return None
    
    # Filter to valid data
    analysis_cols = required_cols + ['est_annual_revenue', 'revenue_per_accommodates']
    available_cols = [col for col in analysis_cols if col in df_analysis.columns]
    valid_data = df_analysis[available_cols].dropna()
    
    print(f"\nUsing {len(valid_data):,} listings with complete data")
    
    # Calculate correlations
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Professionalization metrics
    prof_metrics = ['host_is_professional', 'host_listings_in_city']
    
    # Outcome metrics
    outcome_metrics = ['occupancy_rate', 'price_clean', 'log_price']
    if 'est_annual_revenue' in valid_data.columns:
        outcome_metrics.append('est_annual_revenue')
    if 'revenue_per_accommodates' in valid_data.columns:
        outcome_metrics.append('revenue_per_accommodates')
    
    # Calculate correlations
    correlations = []
    for prof_metric in prof_metrics:
        if prof_metric in valid_data.columns:
            for outcome_metric in outcome_metrics:
                if outcome_metric in valid_data.columns:
                    corr = valid_data[prof_metric].corr(valid_data[outcome_metric])
                    correlations.append({
                        'professionalization_metric': prof_metric,
                        'outcome_metric': outcome_metric,
                        'correlation': corr
                    })
                    print(f"{prof_metric} vs {outcome_metric}: {corr:.4f}")
    
    # Group comparisons
    print(f"\n{'='*80}")
    print(f"GROUP COMPARISONS")
    print(f"{'='*80}")
    
    # Professional vs Casual hosts
    professional = valid_data[valid_data['host_is_professional'] == 1]
    casual = valid_data[valid_data['host_is_professional'] == 0]
    
    print(f"\nProfessional Hosts (2+ listings): {len(professional):,} listings")
    print(f"Casual Hosts (1 listing): {len(casual):,} listings")
    
    for metric in outcome_metrics:
        if metric in valid_data.columns:
            prof_mean = professional[metric].mean()
            casual_mean = casual[metric].mean()
            diff = prof_mean - casual_mean
            pct_diff = (diff / casual_mean * 100) if casual_mean != 0 else 0
            
            print(f"\n{metric}:")
            print(f"  Professional: {prof_mean:.3f}")
            print(f"  Casual: {casual_mean:.3f}")
            print(f"  Difference: {diff:.3f} ({pct_diff:+.1f}%)")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Professional vs Casual - Occupancy
    ax = axes[0, 0]
    plot_data = [casual['occupancy_rate'].dropna(), professional['occupancy_rate'].dropna()]
    bp = ax.boxplot(plot_data, labels=['Casual\n(1 listing)', 'Professional\n(2+ listings)'], 
                   patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_title('Occupancy Rate: Professional vs Casual', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Professional vs Casual - Log Price
    ax = axes[0, 1]
    plot_data = [casual['log_price'].dropna(), professional['log_price'].dropna()]
    bp = ax.boxplot(plot_data, labels=['Casual\n(1 listing)', 'Professional\n(2+ listings)'], 
                   patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax.set_ylabel('Log Price', fontweight='bold', fontsize=11)
    ax.set_title('Log Price: Professional vs Casual', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Host Listings vs Occupancy (scatter)
    ax = axes[0, 2]
    sample_size = min(5000, len(valid_data))
    sample_data = valid_data.sample(n=sample_size, random_state=42)
    ax.scatter(sample_data['host_listings_in_city'], sample_data['occupancy_rate'], 
              alpha=0.3, s=10, edgecolors='none')
    ax.set_xlabel('Host Listings in City', fontweight='bold', fontsize=11)
    ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_title('Host Listings vs Occupancy Rate', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Host Listings vs Log Price (scatter)
    ax = axes[1, 0]
    ax.scatter(sample_data['host_listings_in_city'], sample_data['log_price'], 
              alpha=0.3, s=10, edgecolors='none', color='green')
    ax.set_xlabel('Host Listings in City', fontweight='bold', fontsize=11)
    ax.set_ylabel('Log Price', fontweight='bold', fontsize=11)
    ax.set_title('Host Listings vs Log Price', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Professional Tier vs Occupancy
    ax = axes[1, 1]
    if 'host_professional_tier' in valid_data.columns:
        tier_order = ['casual', 'small_professional', 'medium_professional', 'large_professional']
        tier_data = []
        tier_labels = []
        for tier in tier_order:
            tier_df = valid_data[valid_data['host_professional_tier'] == tier]
            if len(tier_df) > 0:
                tier_data.append(tier_df['occupancy_rate'].dropna())
                tier_labels.append(tier.replace('_', '\n').title())
        
        if tier_data:
            bp = ax.boxplot(tier_data, labels=tier_labels, patch_artist=True)
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
            ax.set_title('Occupancy by Professional Tier', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Revenue comparison (if available)
    ax = axes[1, 2]
    if 'est_annual_revenue' in valid_data.columns:
        plot_data = [casual['est_annual_revenue'].dropna(), professional['est_annual_revenue'].dropna()]
        bp = ax.boxplot(plot_data, labels=['Casual\n(1 listing)', 'Professional\n(2+ listings)'], 
                       patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.7)
        ax.set_ylabel('Estimated Annual Revenue', fontweight='bold', fontsize=11)
        ax.set_title('Annual Revenue: Professional vs Casual', fontweight='bold', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Revenue data\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    
    fig.suptitle(f'Professionalization Correlations: {city_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_professionalization_correlations.png', 
               dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{city_name}_professionalization_correlations.png")
    plt.close()
    
    return correlations


def main():
    """Main execution function"""
    use_detailed = '-all' in sys.argv
    
    # Get city list
    if len(sys.argv) > 1 and sys.argv[1] != '-all':
        cities = [sys.argv[1]]
    else:
        cities = [
            'Albany', 'Asheville', 'Austin', 'Boston', 'Bozeman', 'Cambridge',
            'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
            'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
            'New_York', 'Oakland', 'Portland', 'Paris',
            'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
        ]
    
    print(f"\n{'#'*80}")
    print(f"PROFESSIONALIZATION CORRELATION & OCCUPANCY INVESTIGATION")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\nMODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"\nMODE: SIMPLE ANALYSIS (19 variables)")
        print("WARNING: Need detailed dataset for full analysis")
    
    all_correlations = []
    all_occupancy_stats = []
    
    for city in cities:
        print(f"\n{'='*80}")
        print(f"Processing: {city.upper()}")
        print(f"{'='*80}")
        
        try:
            df = load_city_data(city, base_dir='.', use_detailed=use_detailed)
            
            # Investigate occupancy distribution (optional, disabled by default)
            # Uncomment the following lines to enable occupancy distribution investigation:
            # occ_stats = investigate_occupancy_distribution(df, city)
            # if occ_stats:
            #     occ_stats['city'] = city
            #     all_occupancy_stats.append(occ_stats)
            
            # Analyze professionalization correlations
            correlations = analyze_professionalization_correlations(df, city)
            if correlations:
                for corr in correlations:
                    corr['city'] = city
                    all_correlations.append(corr)
            
        except Exception as e:
            print(f"ERROR processing {city}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary results
    if all_correlations:
        corr_df = pd.DataFrame(all_correlations)
        output_dir = Path('city_comparison_outputs')
        output_dir.mkdir(exist_ok=True)
        corr_df.to_csv(output_dir / 'professionalization_correlations_all_cities.csv', index=False)
        print(f"\n✓ Saved: {output_dir}/professionalization_correlations_all_cities.csv")
    
    if all_occupancy_stats:
        occ_df = pd.DataFrame(all_occupancy_stats)
        output_dir = Path('city_comparison_outputs')
        output_dir.mkdir(exist_ok=True)
        occ_df.to_csv(output_dir / 'occupancy_distribution_stats_all_cities.csv', index=False)
        print(f"✓ Saved: {output_dir}/occupancy_distribution_stats_all_cities.csv")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"OCCUPANCY DISTRIBUTION SUMMARY")
        print(f"{'='*80}")
        print(f"\nMax occupancy rate across all cities: {occ_df['max_rate'].max():.3f} ({occ_df['max_rate'].max()*100:.1f}%)")
        print(f"Mean max occupancy rate: {occ_df['max_rate'].mean():.3f} ({occ_df['max_rate'].mean()*100:.1f}%)")
        print(f"\nCities with max occupancy > 0.7: {(occ_df['max_rate'] > 0.7).sum()}")
        print(f"Cities with max occupancy > 0.8: {(occ_df['max_rate'] > 0.8).sum()}")
        print(f"Cities with max occupancy > 0.9: {(occ_df['max_rate'] > 0.9).sum()}")


if __name__ == "__main__":
    main()

