"""
Market Entry Barrier Analysis for Non-Professional Operators

Creates actionable visualizations to help casual operators (1 listing) evaluate
market entry decisions. Shows mathematical barriers to entry based on professionalization.

Visualizations:
1. Cumulative Profit by Host Listings - Shows what % of total market profit is controlled
   by hosts with ≤N listings
2. Cumulative Revenue by Host Listings - Shows what % of total market revenue is controlled
   by hosts with ≤N listings
3. Performance Gap by Market Professionalization - Shows how the competitive disadvantage
   (professional ROI - casual ROI) grows with market professionalization
4. Casual Host Performance vs Market Average - Shows how casual hosts perform relative
   to market average, highlighting markets where entry is difficult

Usage:
    python analyze_market_entry_barriers.py Austin -all
    python analyze_market_entry_barriers.py -all  # All cities
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

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Import feature engineering functions
from city_level_analysis import apply_all_feature_engineering
from pathlib import Path

def load_city_data_with_features(city_name, base_dir='.', use_detailed=False):
    """Load city data with all feature engineering applied"""
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
    
    # Apply feature engineering (this adds all the required columns)
    df = apply_all_feature_engineering(df, city_name, include_zillow=True)
    
    return df

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def calculate_cumulative_metrics(df, metric_col, listings_col='host_listings_in_city'):
    """
    Calculate cumulative percentage of metric controlled by hosts with ≤N listings.
    
    Args:
        df: DataFrame with listings
        metric_col: Column name for metric to accumulate (e.g., 'annual_cash_flow', 'est_annual_revenue')
        listings_col: Column name for host listings count
        
    Returns:
        DataFrame with columns: listings_threshold, cumulative_pct, cumulative_value
    """
    # Filter to valid data
    valid = df[[metric_col, listings_col]].dropna()
    
    if len(valid) == 0:
        return None
    
    total_metric = valid[metric_col].sum()
    if total_metric == 0:
        return None
    
    # Get unique listing counts (sorted)
    unique_listings = sorted(valid[listings_col].unique())
    
    cumulative_data = []
    for threshold in unique_listings:
        # All hosts with ≤threshold listings
        subset = valid[valid[listings_col] <= threshold]
        cumulative_value = subset[metric_col].sum()
        cumulative_pct = (cumulative_value / total_metric) * 100
        
        cumulative_data.append({
            'listings_threshold': threshold,
            'cumulative_pct': cumulative_pct,
            'cumulative_value': cumulative_value,
            'n_hosts': subset[listings_col].nunique(),
            'n_listings': len(subset)
        })
    
    return pd.DataFrame(cumulative_data)


def analyze_market_entry_barriers(df, city_name, output_dir=None):
    """
    Create market entry barrier analysis visualizations.
    
    Creates a 4-panel visualization:
    1. Cumulative Profit by Host Listings
    2. Cumulative Revenue by Host Listings
    3. Performance Gap by Market Professionalization
    4. Casual Host Performance vs Market Average
    """
    print(f"\n{'='*80}")
    print(f"MARKET ENTRY BARRIER ANALYSIS: {city_name.upper()}")
    print(f"{'='*80}")
    
    if output_dir is None:
        output_dir = Path(city_name) / 'analysis_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exclude hotel rooms
    if 'room_type' in df.columns:
        n_hotel = (df['room_type'] == 'Hotel room').sum()
        if n_hotel > 0:
            df = df[df['room_type'] != 'Hotel room'].copy()
            print(f"  🏨 Excluded {n_hotel:,} hotel room listings")
    
    # Required columns
    required_cols = ['host_listings_in_city', 'host_is_professional', 
                    'cash_on_cash_roi', 'est_annual_revenue', 'annual_cash_flow']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing required columns: {missing_cols}")
        return None
    
    # Filter to valid data
    analysis_cols = required_cols + ['market_professionalization_score']
    available_cols = [col for col in analysis_cols if col in df.columns]
    df_analysis = df[available_cols].dropna()
    
    print(f"  Using {len(df_analysis):,} listings with complete data")
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ============================================================================
    # PANEL 1: Cumulative Profit by Host Listings
    # ============================================================================
    ax1 = axes[0, 0]
    
    # Calculate cumulative profit
    cum_profit = calculate_cumulative_metrics(df_analysis, 'annual_cash_flow', 'host_listings_in_city')
    
    if cum_profit is not None and len(cum_profit) > 0:
        # Plot cumulative percentage
        ax1.plot(cum_profit['listings_threshold'], cum_profit['cumulative_pct'], 
                'b-', linewidth=2.5, marker='o', markersize=4, label='Cumulative % of Profit')
        
        # Add reference lines
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        
        # Find key thresholds
        p50_threshold = cum_profit[cum_profit['cumulative_pct'] >= 50]['listings_threshold'].min()
        p80_threshold = cum_profit[cum_profit['cumulative_pct'] >= 80]['listings_threshold'].min()
        
        if pd.notna(p50_threshold):
            ax1.axvline(x=p50_threshold, color='orange', linestyle='--', alpha=0.3)
            ax1.text(p50_threshold, 50, f'  {p50_threshold:.0f} listings\n  control 50%', 
                    fontsize=9, va='bottom', ha='left')
        
        if pd.notna(p80_threshold):
            ax1.axvline(x=p80_threshold, color='red', linestyle='--', alpha=0.3)
            ax1.text(p80_threshold, 80, f'  {p80_threshold:.0f} listings\n  control 80%', 
                    fontsize=9, va='bottom', ha='left')
        
        ax1.set_xlabel('Host Listings in City (≤N)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Cumulative % of Total Market Profit', fontweight='bold', fontsize=12)
        ax1.set_title('1. Cumulative Profit by Host Listings\n(What % of profit is controlled by hosts with ≤N listings)', 
                     fontweight='bold', fontsize=13)
        ax1.set_xlim(0, min(cum_profit['listings_threshold'].max(), 50))  # Focus on 0-50 range
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=10)
        
        # Add interpretation text
        if p50_threshold and p50_threshold <= 5:
            interpretation = "⚠️ High concentration: Small operators control most profit"
            color = 'red'
        elif p50_threshold and p50_threshold <= 10:
            interpretation = "⚠️ Moderate concentration: Entry may be difficult"
            color = 'orange'
        else:
            interpretation = "✓ Lower concentration: More accessible market"
            color = 'green'
        
        ax1.text(0.02, 0.98, interpretation, transform=ax1.transAxes,
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ============================================================================
    # PANEL 2: Cumulative Revenue by Host Listings
    # ============================================================================
    ax2 = axes[0, 1]
    
    # Calculate cumulative revenue
    cum_revenue = calculate_cumulative_metrics(df_analysis, 'est_annual_revenue', 'host_listings_in_city')
    
    if cum_revenue is not None and len(cum_revenue) > 0:
        # Plot cumulative percentage
        ax2.plot(cum_revenue['listings_threshold'], cum_revenue['cumulative_pct'], 
                'g-', linewidth=2.5, marker='s', markersize=4, label='Cumulative % of Revenue')
        
        # Add reference lines
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        
        # Find key thresholds
        p50_threshold = cum_revenue[cum_revenue['cumulative_pct'] >= 50]['listings_threshold'].min()
        p80_threshold = cum_revenue[cum_revenue['cumulative_pct'] >= 80]['listings_threshold'].min()
        
        if pd.notna(p50_threshold):
            ax2.axvline(x=p50_threshold, color='orange', linestyle='--', alpha=0.3)
            ax2.text(p50_threshold, 50, f'  {p50_threshold:.0f} listings\n  control 50%', 
                    fontsize=9, va='bottom', ha='left')
        
        if pd.notna(p80_threshold):
            ax2.axvline(x=p80_threshold, color='red', linestyle='--', alpha=0.3)
            ax2.text(p80_threshold, 80, f'  {p80_threshold:.0f} listings\n  control 80%', 
                    fontsize=9, va='bottom', ha='left')
        
        ax2.set_xlabel('Host Listings in City (≤N)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Cumulative % of Total Market Revenue', fontweight='bold', fontsize=12)
        ax2.set_title('2. Cumulative Revenue by Host Listings\n(What % of revenue is controlled by hosts with ≤N listings)', 
                     fontweight='bold', fontsize=13)
        ax2.set_xlim(0, min(cum_revenue['listings_threshold'].max(), 50))
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)
        
        # Add interpretation
        if p50_threshold and p50_threshold <= 5:
            interpretation = "⚠️ High concentration: Small operators control most revenue"
            color = 'red'
        elif p50_threshold and p50_threshold <= 10:
            interpretation = "⚠️ Moderate concentration: Entry may be difficult"
            color = 'orange'
        else:
            interpretation = "✓ Lower concentration: More accessible market"
            color = 'green'
        
        ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes,
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ============================================================================
    # PANEL 3: Performance Gap by Market Professionalization
    # ============================================================================
    ax3 = axes[1, 0]
    
    # Calculate performance gap (professional ROI - casual ROI)
    # For this single-city analysis, we'll show the gap as a function of market professionalization
    # Since we only have one city, we'll show the gap directly
    
    professional = df_analysis[df_analysis['host_is_professional'] == 1]
    casual = df_analysis[df_analysis['host_is_professional'] == 0]
    
    if len(professional) > 0 and len(casual) > 0:
        prof_roi = professional['cash_on_cash_roi'].median()
        casual_roi = casual['cash_on_cash_roi'].median()
        performance_gap = prof_roi - casual_roi
        
        # Get market professionalization score
        if 'market_professionalization_score' in df_analysis.columns:
            prof_score = df_analysis['market_professionalization_score'].iloc[0]
        else:
            # Calculate it
            pct_prof = (len(professional) / len(df_analysis)) * 100
            prof_score = pct_prof  # Simplified
        
        # Create scatter plot (for single city, this is a single point)
        # But we can show it with context
        ax3.scatter(prof_score, performance_gap, s=500, alpha=0.7, 
                   color='red' if performance_gap > 0 else 'green', 
                   edgecolors='black', linewidth=2, zorder=5)
        
        # Add reference lines
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% professionalization')
        
        # Add annotation
        ax3.annotate(f'{city_name}\nGap: {performance_gap:.1f}%\nProf Score: {prof_score:.1f}',
                    xy=(prof_score, performance_gap),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Add interpretation zones
        ax3.axhspan(-100, 0, alpha=0.1, color='green', label='Casual advantage')
        ax3.axhspan(0, 100, alpha=0.1, color='red', label='Professional advantage')
        
        ax3.set_xlabel('Market Professionalization Score (0-100)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Performance Gap (Professional ROI - Casual ROI, %)', fontweight='bold', fontsize=12)
        ax3.set_title('3. Performance Gap by Market Professionalization\n(Higher gap = harder for casual operators)', 
                     fontweight='bold', fontsize=13)
        ax3.set_xlim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        # Add interpretation
        if performance_gap > 10 and prof_score > 50:
            interpretation = "⚠️ High barrier: Large gap in professionalized market"
            color = 'red'
        elif performance_gap > 5:
            interpretation = "⚠️ Moderate barrier: Professional advantage exists"
            color = 'orange'
        else:
            interpretation = "✓ Lower barrier: Casual operators competitive"
            color = 'green'
        
        ax3.text(0.02, 0.98, interpretation, transform=ax3.transAxes,
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ============================================================================
    # PANEL 4: Casual Host Performance vs Market Average
    # ============================================================================
    ax4 = axes[1, 1]
    
    if len(casual) > 0:
        # Calculate metrics for casual hosts and market average
        casual_roi = casual['cash_on_cash_roi'].median()
        market_roi = df_analysis['cash_on_cash_roi'].median()
        
        casual_revenue = casual['est_annual_revenue'].median()
        market_revenue = df_analysis['est_annual_revenue'].median()
        
        casual_occupancy = casual['occupancy_rate'].median() if 'occupancy_rate' in casual.columns else np.nan
        market_occupancy = df_analysis['occupancy_rate'].median() if 'occupancy_rate' in df_analysis.columns else np.nan
        
        # Create comparison bars
        metrics = ['ROI (%)', 'Revenue ($)', 'Occupancy']
        casual_values = [casual_roi, casual_revenue/1000, casual_occupancy*100 if pd.notna(casual_occupancy) else np.nan]
        market_values = [market_roi, market_revenue/1000, market_occupancy*100 if pd.notna(market_occupancy) else np.nan]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, casual_values, width, label='Casual Hosts (1 listing)', 
                       color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax4.bar(x + width/2, market_values, width, label='Market Average', 
                       color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height) and not np.isinf(height):
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}' if abs(height) < 1000 else f'{height:.0f}',
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9, fontweight='bold')
        
        # Calculate performance ratio
        if market_roi != 0:
            roi_ratio = (casual_roi / market_roi) * 100
        else:
            roi_ratio = 0
        
        ax4.set_xlabel('Metric', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax4.set_title(f'4. Casual Host Performance vs Market Average\n(Casual ROI = {roi_ratio:.0f}% of market average)', 
                     fontweight='bold', fontsize=13)
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(loc='best', fontsize=10)
        
        # Add interpretation
        if roi_ratio < 70:
            interpretation = "⚠️ Casual hosts significantly underperform"
            color = 'red'
        elif roi_ratio < 90:
            interpretation = "⚠️ Casual hosts moderately underperform"
            color = 'orange'
        else:
            interpretation = "✓ Casual hosts competitive with market"
            color = 'green'
        
        ax4.text(0.02, 0.98, interpretation, transform=ax4.transAxes,
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Overall title
    fig.suptitle(f'Market Entry Barrier Analysis: {city_name}\n(Actionable Insights for Non-Professional Operators)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{city_name}_market_entry_barriers.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"MARKET ENTRY BARRIER SUMMARY")
    print(f"{'='*80}")
    
    if len(professional) > 0 and len(casual) > 0:
        print(f"\nProfessional Hosts (2+ listings): {len(professional):,} listings")
        print(f"Casual Hosts (1 listing): {len(casual):,} listings")
        print(f"\nPerformance Comparison:")
        print(f"  Professional median ROI: {prof_roi:.1f}%")
        print(f"  Casual median ROI: {casual_roi:.1f}%")
        print(f"  Performance gap: {performance_gap:.1f} percentage points")
        print(f"\nMarket Concentration:")
        if cum_profit is not None and len(cum_profit) > 0:
            p50_threshold = cum_profit[cum_profit['cumulative_pct'] >= 50]['listings_threshold'].min()
            p80_threshold = cum_profit[cum_profit['cumulative_pct'] >= 80]['listings_threshold'].min()
            if pd.notna(p50_threshold):
                print(f"  50% of profit controlled by hosts with ≤{p50_threshold:.0f} listings")
            if pd.notna(p80_threshold):
                print(f"  80% of profit controlled by hosts with ≤{p80_threshold:.0f} listings")
    
    return output_file


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
    
    if not city_args:
        print("Usage: python analyze_market_entry_barriers.py CityName [-all]")
        print("       python analyze_market_entry_barriers.py -all  # All cities")
        print("\nExample: python analyze_market_entry_barriers.py Austin -all")
        sys.exit(1)
    
    if city_args[0] == '-all' or city_args[0].lower() == 'all':
        city_folders = all_cities
        print(f"\n📊 BATCH MODE: Analyzing all {len(city_folders)} cities")
    else:
        city_folders = [city_args[0]]
        print(f"\n🎯 SINGLE CITY MODE: Analyzing {city_folders[0]} only")
    
    if use_detailed:
        print(f"🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"🔍 MODE: SIMPLE ANALYSIS (19 variables)")
    
    results = []
    
    for city_name in city_folders:
        print(f"\n{'='*80}")
        print(f"Processing: {city_name}")
        print(f"{'='*80}")
        
        try:
            df = load_city_data_with_features(city_name, base_dir='.', use_detailed=use_detailed)
            result = analyze_market_entry_barriers(df, city_name)
            if result:
                results.append((city_name, result))
        except Exception as e:
            print(f"  ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nSuccessfully analyzed {len(results)} cities:")
    for city, result in results:
        print(f"  - {city}: {result}")
    
    return results


if __name__ == "__main__":
    main()

