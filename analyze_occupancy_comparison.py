"""
Occupancy Metrics Comparison Analysis
Compares occupancy_proxy (calendar-based) vs estimated_occupancy_l365d (actual bookings)
to quantify the host blocking effect

Usage:
    python analyze_occupancy_comparison.py Austin -all
    python analyze_occupancy_comparison.py -all  # All cities
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
plt.rcParams['figure.figsize'] = (14, 10)


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


def analyze_occupancy_comparison(df, city_name, output_dir=None):
    """
    Compare occupancy_proxy vs estimated_occupancy_l365d to quantify host blocking
    
    Args:
        df: DataFrame with feature engineering applied
        city_name: Name of city
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*80}")
    print(f"OCCUPANCY METRICS COMPARISON FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path('.') / city_name / 'analysis_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate both metrics
    df_analysis = df.copy()
    
    # Calendar-based proxy (includes host blocks)
    if 'availability_365' in df_analysis.columns:
        df_analysis['calendar_occupancy_proxy'] = (365 - df_analysis['availability_365']) / 365
        df_analysis['unavailable_days_calendar'] = 365 - df_analysis['availability_365']
    
    # Actual booked days (from Airbnb)
    if 'estimated_occupancy_l365d' in df_analysis.columns:
        df_analysis['actual_occupancy_rate'] = df_analysis['estimated_occupancy_l365d'] / 365
        df_analysis['booked_days'] = df_analysis['estimated_occupancy_l365d']
    else:
        print("WARNING: estimated_occupancy_l365d not found in data")
        print("Cannot perform comparison - need detailed dataset with estimated_occupancy_l365d")
        return
    
    # Calculate host-blocked days (difference)
    if 'unavailable_days_calendar' in df_analysis.columns and 'booked_days' in df_analysis.columns:
        df_analysis['host_blocked_days'] = df_analysis['unavailable_days_calendar'] - df_analysis['booked_days']
        df_analysis['host_blocked_rate'] = df_analysis['host_blocked_days'] / 365
    
    # Filter to valid comparisons
    comparison_cols = ['calendar_occupancy_proxy', 'actual_occupancy_rate']
    if 'host_blocked_rate' in df_analysis.columns:
        comparison_cols.extend(['host_blocked_rate', 'host_blocked_days'])
    comparison_cols.append('price_clean')
    valid_data = df_analysis[comparison_cols].dropna()
    
    print(f"\nUsing {len(valid_data):,} listings with complete occupancy data")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nCalendar-Based Occupancy Proxy (includes host blocks):")
    print(f"  Mean: {df_analysis['calendar_occupancy_proxy'].mean():.3f} ({df_analysis['calendar_occupancy_proxy'].mean()*100:.1f}%)")
    print(f"  Median: {df_analysis['calendar_occupancy_proxy'].median():.3f} ({df_analysis['calendar_occupancy_proxy'].median()*100:.1f}%)")
    print(f"  Std: {df_analysis['calendar_occupancy_proxy'].std():.3f}")
    
    print(f"\nActual Occupancy Rate (booked days only):")
    print(f"  Mean: {df_analysis['actual_occupancy_rate'].mean():.3f} ({df_analysis['actual_occupancy_rate'].mean()*100:.1f}%)")
    print(f"  Median: {df_analysis['actual_occupancy_rate'].median():.3f} ({df_analysis['actual_occupancy_rate'].median()*100:.1f}%)")
    print(f"  Std: {df_analysis['actual_occupancy_rate'].std():.3f}")
    
    if 'host_blocked_rate' in df_analysis.columns:
        print(f"\nHost-Blocked Rate (difference):")
        print(f"  Mean: {df_analysis['host_blocked_rate'].mean():.3f} ({df_analysis['host_blocked_rate'].mean()*100:.1f}%)")
        print(f"  Median: {df_analysis['host_blocked_rate'].median():.3f} ({df_analysis['host_blocked_rate'].median()*100:.1f}%)")
        print(f"  Std: {df_analysis['host_blocked_rate'].std():.3f}")
        print(f"  Mean blocked days: {df_analysis['host_blocked_days'].mean():.1f} days/year")
    
    # Calculate overestimation factor
    if len(valid_data) > 0:
        mean_calendar = valid_data['calendar_occupancy_proxy'].mean()
        mean_actual = valid_data['actual_occupancy_rate'].mean()
        if mean_actual > 0:
            overestimation_factor = mean_calendar / mean_actual
            print(f"\n{'='*80}")
            print(f"OVERESTIMATION ANALYSIS")
            print(f"{'='*80}")
            print(f"Calendar proxy overestimates actual occupancy by: {overestimation_factor:.2f}x")
            print(f"  Calendar proxy: {mean_calendar*100:.1f}%")
            print(f"  Actual occupancy: {mean_actual*100:.1f}%")
            print(f"  Difference: {(mean_calendar - mean_actual)*100:.1f} percentage points")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Figure 1: Scatter plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Calendar vs Actual Occupancy
    ax = axes[0, 0]
    ax.scatter(valid_data['calendar_occupancy_proxy'], valid_data['actual_occupancy_rate'], 
              alpha=0.4, s=20, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
    ax.set_xlabel('Calendar Occupancy Proxy (includes blocks)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual Occupancy Rate (booked only)', fontweight='bold', fontsize=11)
    ax.set_title('Calendar Proxy vs Actual Occupancy', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Calculate correlation
    corr = valid_data['calendar_occupancy_proxy'].corr(valid_data['actual_occupancy_rate'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution comparison
    ax = axes[0, 1]
    ax.hist(valid_data['calendar_occupancy_proxy'], bins=50, alpha=0.6, label='Calendar Proxy', density=True)
    ax.hist(valid_data['actual_occupancy_rate'], bins=50, alpha=0.6, label='Actual Occupancy', density=True)
    ax.set_xlabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_ylabel('Density', fontweight='bold', fontsize=11)
    ax.set_title('Distribution Comparison', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Host-blocked rate distribution
    if 'host_blocked_rate' in valid_data.columns:
        ax = axes[1, 0]
        ax.hist(valid_data['host_blocked_rate'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Host-Blocked Rate', fontweight='bold', fontsize=11)
        ax.set_ylabel('Count', fontweight='bold', fontsize=11)
        ax.set_title(f'Host-Blocked Days Distribution\n(Mean: {valid_data["host_blocked_rate"].mean()*100:.1f}%)', 
                    fontweight='bold', fontsize=12)
        ax.axvline(valid_data['host_blocked_rate'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {valid_data["host_blocked_rate"].mean()*100:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Box plot comparison
    ax = axes[1, 1]
    box_data = [valid_data['calendar_occupancy_proxy'], valid_data['actual_occupancy_rate']]
    bp = ax.boxplot(box_data, labels=['Calendar Proxy', 'Actual Occupancy'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Occupancy Rate', fontweight='bold', fontsize=11)
    ax.set_title('Occupancy Metrics Comparison (Box Plot)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)
    
    fig.suptitle(f'{city_name.upper()} - Occupancy Metrics Comparison Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_occupancy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / city_name}_occupancy_comparison.png")
    plt.close()
    
    # Save summary statistics to CSV
    summary_stats = {
        'metric': ['calendar_occupancy_proxy', 'actual_occupancy_rate', 'host_blocked_rate'],
        'mean': [
            df_analysis['calendar_occupancy_proxy'].mean(),
            df_analysis['actual_occupancy_rate'].mean(),
            df_analysis['host_blocked_rate'].mean() if 'host_blocked_rate' in df_analysis.columns else None
        ],
        'median': [
            df_analysis['calendar_occupancy_proxy'].median(),
            df_analysis['actual_occupancy_rate'].median(),
            df_analysis['host_blocked_rate'].median() if 'host_blocked_rate' in df_analysis.columns else None
        ],
        'std': [
            df_analysis['calendar_occupancy_proxy'].std(),
            df_analysis['actual_occupancy_rate'].std(),
            df_analysis['host_blocked_rate'].std() if 'host_blocked_rate' in df_analysis.columns else None
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / f'{city_name}_occupancy_comparison_stats.csv', index=False)
    print(f"✓ Saved: {output_dir / city_name}_occupancy_comparison_stats.csv")
    
    # Save detailed comparison data
    save_cols = ['calendar_occupancy_proxy', 'actual_occupancy_rate']
    if 'host_blocked_rate' in valid_data.columns:
        save_cols.extend(['host_blocked_rate', 'host_blocked_days'])
    comparison_data = valid_data[save_cols].copy()
    comparison_data.to_csv(output_dir / f'{city_name}_occupancy_comparison_data.csv', index=False)
    print(f"✓ Saved: {output_dir / city_name}_occupancy_comparison_data.csv")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")


def main():
    """Main execution function"""
    use_detailed = '-all' in sys.argv
    
    # Get city name (first non-flag argument)
    city_args = [arg for arg in sys.argv[1:] if arg != '-all']
    
    # List of all cities
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    
    # Check if user wants all cities (either -all is the only arg, or it's in the args)
    if '-all' in sys.argv and len(city_args) == 0:
        # User ran: python analyze_occupancy_comparison.py -all
        city_name = None
    elif city_args:
        # User specified a city name
        city_name = city_args[0]
    else:
        print("Usage: python analyze_occupancy_comparison.py CityName [-all]")
        print("       python analyze_occupancy_comparison.py -all  # All cities")
        print("\nExample: python analyze_occupancy_comparison.py Austin -all")
        sys.exit(1)
    
    if city_name:
        # Single city analysis
        print(f"\n{'#'*80}")
        print(f"OCCUPANCY COMPARISON ANALYSIS - {city_name.upper()}")
        print(f"{'#'*80}")
        
        try:
            df = load_city_data(city_name, base_dir='.', use_detailed=use_detailed)
            analyze_occupancy_comparison(df, city_name)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # All cities analysis
        print(f"\n{'#'*80}")
        print(f"OCCUPANCY COMPARISON ANALYSIS - ALL CITIES")
        print(f"{'#'*80}")
        
        results = []
        successful = 0
        failed = 0
        
        for city in all_cities:
            print(f"\n{'='*80}")
            print(f"Processing: {city.upper()}")
            print(f"{'='*80}")
            
            try:
                df = load_city_data(city, base_dir='.', use_detailed=use_detailed)
                
                # Quick analysis to extract key metrics
                df_analysis = df.copy()
                
                if 'availability_365' in df_analysis.columns:
                    df_analysis['calendar_occupancy_proxy'] = (365 - df_analysis['availability_365']) / 365
                
                if 'estimated_occupancy_l365d' in df_analysis.columns:
                    df_analysis['actual_occupancy_rate'] = df_analysis['estimated_occupancy_l365d'] / 365
                    
                    if 'calendar_occupancy_proxy' in df_analysis.columns:
                        df_analysis['host_blocked_rate'] = (
                            df_analysis['calendar_occupancy_proxy'] - df_analysis['actual_occupancy_rate']
                        )
                        
                        # Get valid data
                        valid = df_analysis[['calendar_occupancy_proxy', 'actual_occupancy_rate', 
                                            'host_blocked_rate']].dropna()
                        
                        if len(valid) > 0:
                            result = {
                                'city': city,
                                'n_listings': len(valid),
                                'calendar_occ_mean': valid['calendar_occupancy_proxy'].mean(),
                                'calendar_occ_median': valid['calendar_occupancy_proxy'].median(),
                                'actual_occ_mean': valid['actual_occupancy_rate'].mean(),
                                'actual_occ_median': valid['actual_occupancy_rate'].median(),
                                'host_blocked_mean': valid['host_blocked_rate'].mean(),
                                'host_blocked_median': valid['host_blocked_rate'].median(),
                                'overestimation_factor': valid['calendar_occupancy_proxy'].mean() / valid['actual_occupancy_rate'].mean() if valid['actual_occupancy_rate'].mean() > 0 else None
                            }
                            results.append(result)
                            successful += 1
                            
                            print(f"✓ {city}: Calendar={result['calendar_occ_mean']*100:.1f}%, "
                                f"Actual={result['actual_occ_mean']*100:.1f}%, "
                                f"Blocked={result['host_blocked_mean']*100:.1f}%, "
                                f"Overestimate={result['overestimation_factor']:.2f}x")
                        else:
                            print(f"⚠️  {city}: No valid data for comparison")
                            failed += 1
                    else:
                        print(f"⚠️  {city}: Missing availability_365")
                        failed += 1
                else:
                    print(f"⚠️  {city}: Missing estimated_occupancy_l365d (need -all flag)")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ ERROR processing {city}: {e}")
                failed += 1
        
        # Create summary DataFrame
        if results:
            summary_df = pd.DataFrame(results)
            summary_df = summary_df.sort_values('host_blocked_mean', ascending=False)
            
            # Save summary
            output_file = Path('city_comparison_outputs') / 'occupancy_comparison_all_cities.csv'
            output_file.parent.mkdir(exist_ok=True)
            summary_df.to_csv(output_file, index=False)
            
            print(f"\n{'#'*80}")
            print(f"SUMMARY - ALL CITIES")
            print(f"{'#'*80}")
            print(f"\nSuccessfully analyzed: {successful} cities")
            print(f"Failed: {failed} cities")
            print(f"\nSaved summary to: {output_file}")
            
            print(f"\n{'='*80}")
            print(f"HOST-BLOCKING RATE BY CITY (sorted by highest blocking)")
            print(f"{'='*80}")
            print(summary_df[['city', 'n_listings', 'calendar_occ_mean', 'actual_occ_mean', 
                            'host_blocked_mean', 'overestimation_factor']].to_string(index=False))
            
            print(f"\n{'='*80}")
            print(f"AGGREGATE STATISTICS")
            print(f"{'='*80}")
            print(f"Mean host-blocked rate across all cities: {summary_df['host_blocked_mean'].mean()*100:.1f}%")
            print(f"Median host-blocked rate: {summary_df['host_blocked_mean'].median()*100:.1f}%")
            print(f"Std dev: {summary_df['host_blocked_mean'].std()*100:.1f}%")
            print(f"\nMean overestimation factor: {summary_df['overestimation_factor'].mean():.2f}x")
            print(f"Median overestimation factor: {summary_df['overestimation_factor'].median():.2f}x")
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Host-blocked rate by city
            ax = axes[0, 0]
            summary_df_sorted = summary_df.sort_values('host_blocked_mean', ascending=True)
            ax.barh(range(len(summary_df_sorted)), summary_df_sorted['host_blocked_mean'] * 100)
            ax.set_yticks(range(len(summary_df_sorted)))
            ax.set_yticklabels(summary_df_sorted['city'], fontsize=9)
            ax.set_xlabel('Host-Blocked Rate (%)', fontweight='bold')
            ax.set_title('Host-Blocked Days by City', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Overestimation factor by city
            ax = axes[0, 1]
            summary_df_sorted2 = summary_df.sort_values('overestimation_factor', ascending=True)
            ax.barh(range(len(summary_df_sorted2)), summary_df_sorted2['overestimation_factor'])
            ax.set_yticks(range(len(summary_df_sorted2)))
            ax.set_yticklabels(summary_df_sorted2['city'], fontsize=9)
            ax.set_xlabel('Overestimation Factor (x)', fontweight='bold')
            ax.set_title('Calendar Proxy Overestimation by City', fontweight='bold', fontsize=12)
            ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='No Overestimation')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Scatter: Calendar vs Actual
            ax = axes[1, 0]
            ax.scatter(summary_df['calendar_occ_mean'] * 100, summary_df['actual_occ_mean'] * 100, 
                      s=100, alpha=0.6, edgecolors='black', linewidth=1)
            for _, row in summary_df.iterrows():
                ax.annotate(row['city'], 
                          (row['calendar_occ_mean'] * 100, row['actual_occ_mean'] * 100),
                          fontsize=8, alpha=0.7)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Agreement')
            ax.set_xlabel('Calendar Occupancy Proxy (%)', fontweight='bold')
            ax.set_ylabel('Actual Occupancy Rate (%)', fontweight='bold')
            ax.set_title('Calendar Proxy vs Actual Occupancy (by City)', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Distribution of host-blocked rates
            ax = axes[1, 1]
            ax.hist(summary_df['host_blocked_mean'] * 100, bins=15, alpha=0.7, edgecolor='black')
            ax.axvline(summary_df['host_blocked_mean'].mean() * 100, color='red', 
                     linestyle='--', linewidth=2, label=f"Mean: {summary_df['host_blocked_mean'].mean()*100:.1f}%")
            ax.set_xlabel('Host-Blocked Rate (%)', fontweight='bold')
            ax.set_ylabel('Number of Cities', fontweight='bold')
            ax.set_title('Distribution of Host-Blocking Rates Across Cities', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            fig.suptitle('Occupancy Metrics Comparison - All Cities', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(output_file.parent / 'occupancy_comparison_all_cities.png', 
                       dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved visualization: {output_file.parent / 'occupancy_comparison_all_cities.png'}")
            plt.close()


if __name__ == "__main__":
    main()

