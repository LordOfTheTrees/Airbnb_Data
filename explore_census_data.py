"""
Census Data Exploration
Explores relationships between metropolitan area population and Airbnb market characteristics

Usage:
    python explore_census_data.py Austin  # Single city
    python explore_census_data.py -all    # All cities
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

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_census_data(census_file='Census/cbsa-est2024-alldata.csv', base_dir='.'):
    """
    Load census metropolitan area data.
    
    Returns:
        DataFrame with CBSA-level data (metropolitan statistical areas)
    """
    # Handle both relative and absolute paths
    if Path(census_file).is_absolute():
        census_path = Path(census_file)
    else:
        census_path = Path(base_dir) / census_file
    
    # Try alternative paths if not found
    if not census_path.exists():
        # Try with different separators
        alt_path = Path(base_dir) / 'Census' / 'cbsa-est2024-alldata.csv'
        if alt_path.exists():
            census_path = alt_path
        else:
            raise FileNotFoundError(f"Census file not found: {census_path} or {alt_path}")
    
    print(f"Loading census data from: {census_path}")
    # Try different encodings
    try:
        df = pd.read_csv(census_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(census_path, encoding='latin-1')
        except:
            df = pd.read_csv(census_path, encoding='cp1252')
    
    # Filter to metropolitan statistical areas (MSAs) only
    # LSAD = "Metropolitan Statistical Area"
    df_msa = df[df['LSAD'] == 'Metropolitan Statistical Area'].copy()
    
    print(f"  Loaded {len(df_msa):,} metropolitan statistical areas")
    
    # Extract key columns
    # Use 2024 estimates (most recent)
    key_cols = ['CBSA', 'NAME', 'POPESTIMATE2024', 'POPESTIMATE2023', 
                'POPESTIMATE2022', 'POPESTIMATE2021', 'POPESTIMATE2020']
    
    available_cols = [col for col in key_cols if col in df_msa.columns]
    df_msa = df_msa[available_cols].copy()
    
    # Rename for clarity
    if 'POPESTIMATE2024' in df_msa.columns:
        df_msa['population_2024'] = df_msa['POPESTIMATE2024']
    if 'POPESTIMATE2023' in df_msa.columns:
        df_msa['population_2023'] = df_msa['POPESTIMATE2023']
    
    # Calculate population growth rate
    if 'POPESTIMATE2024' in df_msa.columns and 'POPESTIMATE2020' in df_msa.columns:
        df_msa['pop_growth_4yr'] = ((df_msa['POPESTIMATE2024'] - df_msa['POPESTIMATE2020']) / 
                                   df_msa['POPESTIMATE2020']) * 100
    
    return df_msa


def match_city_to_census(city_name, census_df):
    """
    Match Airbnb city name to census metropolitan area.
    
    Args:
        city_name: Airbnb city name (e.g., "Austin")
        census_df: Census DataFrame with NAME column
        
    Returns:
        Matching census row or None
    """
    # Common mappings
    city_mappings = {
        'Austin': ['Austin', 'Austin-Round Rock'],
        'Chicago': ['Chicago', 'Chicago-Naperville'],
        'Dallas': ['Dallas', 'Dallas-Fort Worth'],
        'Denver': ['Denver', 'Denver-Aurora'],
        'Los_Angeles': ['Los Angeles', 'Los Angeles-Long Beach'],
        'New_York': ['New York', 'New York-Newark'],
        'San_Francisco': ['San Francisco', 'San Francisco-Oakland'],
        'Seattle': ['Seattle', 'Seattle-Tacoma'],
        'Washington_DC': ['Washington', 'Washington-Arlington'],
        'Nashville': ['Nashville', 'Nashville-Davidson'],
        'New_Orleans': ['New Orleans', 'New Orleans-Metairie'],
        'Jersey_City': ['New York', 'New York-Newark'],  # Part of NYC metro
        'Cambridge': ['Boston', 'Boston-Cambridge'],  # Part of Boston metro
        'Oakland': ['San Francisco', 'San Francisco-Oakland'],  # Part of SF metro
        'Albany': ['Albany', 'Albany-Schenectady'],
        'Asheville': ['Asheville'],
        'Bozeman': ['Bozeman'],
        'Columbus': ['Columbus'],
        'Hawaii': ['Hawaii', 'Honolulu'],  # Hawaii, Hawaii or Honolulu metro
        'Portland': ['Portland', 'Portland-Vancouver'],  # Portland, Oregon
        'Paris': None,  # Not US
        'Rhode_Island': ['Providence', 'Providence-Warwick'],  # Rhode Island metro area
    }
    
    # Try direct match first
    if city_name in city_mappings:
        search_terms = city_mappings[city_name]
        if search_terms is None:
            return None
    else:
        # Try to find city name in census names
        search_terms = [city_name.replace('_', ' '), city_name]
    
    # Search for matching MSA
    for term in search_terms:
        matches = census_df[census_df['NAME'].str.contains(term, case=False, na=False)]
        if len(matches) > 0:
            # Return the largest MSA if multiple matches
            if 'population_2024' in matches.columns:
                return matches.nlargest(1, 'population_2024').iloc[0]
            else:
                return matches.iloc[0]
    
    return None


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
    df = apply_all_feature_engineering(df, city_name, include_zillow=True)
    
    return df


def explore_census_relationships(city_name, df, census_data, output_dir=None):
    """
    Explore relationships between census population data and Airbnb metrics.
    
    Args:
        city_name: Name of city
        df: DataFrame with Airbnb listing data (feature engineered)
        census_data: Census DataFrame row for this city's metro area
        output_dir: Directory to save outputs
    """
    if output_dir is None:
        output_dir = Path(city_name) / 'exploration_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CENSUS DATA EXPLORATION: {city_name.upper()}")
    print(f"{'='*80}")
    
    if census_data is None:
        print(f"  ⚠️  No census data match found for {city_name}")
        return None
    
    # Extract population info
    pop_2024 = census_data.get('population_2024', np.nan)
    pop_growth = census_data.get('pop_growth_4yr', np.nan)
    msa_name = census_data.get('NAME', 'Unknown')
    
    print(f"  Metropolitan Area: {msa_name}")
    if pd.notna(pop_2024):
        print(f"  Population (2024): {pop_2024:,.0f}")
    if pd.notna(pop_growth):
        print(f"  Population Growth (2020-2024): {pop_growth:.2f}%")
    
    # Calculate city-level metrics
    metrics = {}
    
    # Market size
    metrics['total_listings'] = len(df)
    
    # Pricing
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
        metrics['mean_roi'] = df['cash_on_cash_roi'].mean()
    
    # Professionalization
    if 'host_is_professional' in df.columns:
        metrics['pct_professional'] = (df['host_is_professional'].sum() / len(df)) * 100
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Population vs Market Size (listings)
    ax = axes[0, 0]
    if pd.notna(pop_2024):
        ax.scatter(pop_2024, metrics['total_listings'], s=300, alpha=0.7, 
                  color='steelblue', edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(city_name, (pop_2024, metrics['total_listings']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Airbnb Listings', fontweight='bold', fontsize=12)
        ax.set_title('Population vs Market Size', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Plot 2: Population vs Median Price
    ax = axes[0, 1]
    if pd.notna(pop_2024) and 'median_price' in metrics:
        ax.scatter(pop_2024, metrics['median_price'], s=300, alpha=0.7,
                  color='coral', edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(city_name, (pop_2024, metrics['median_price']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Median Nightly Price ($)', fontweight='bold', fontsize=12)
        ax.set_title('Population vs Median Price', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Plot 3: Population vs Median Occupancy
    ax = axes[1, 0]
    if pd.notna(pop_2024) and 'median_occupancy' in metrics:
        ax.scatter(pop_2024, metrics['median_occupancy'] * 100, s=300, alpha=0.7,
                  color='green', edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(city_name, (pop_2024, metrics['median_occupancy'] * 100),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Median Occupancy Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Population vs Occupancy Rate', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Plot 4: Population vs ROI
    ax = axes[1, 1]
    if pd.notna(pop_2024) and 'median_roi' in metrics:
        ax.scatter(pop_2024, metrics['median_roi'], s=300, alpha=0.7,
                  color='purple', edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(city_name, (pop_2024, metrics['median_roi']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Median Cash-on-Cash ROI (%)', fontweight='bold', fontsize=12)
        ax.set_title('Population vs ROI', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig.suptitle(f'Census Data Exploration: {city_name}\n{msa_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'{city_name}_census_exploration.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Print summary
    print(f"\n  City Metrics:")
    for key, value in metrics.items():
        if pd.notna(value):
            if 'price' in key or 'revenue' in key:
                print(f"    {key}: ${value:,.0f}")
            elif 'occupancy' in key:
                print(f"    {key}: {value*100:.1f}%")
            elif 'roi' in key or 'pct' in key:
                print(f"    {key}: {value:.1f}%")
            else:
                print(f"    {key}: {value:,.0f}")
    
    return metrics


def create_all_cities_census_visualization(city_results, output_dir=None):
    """
    Create a quad visualization showing all US cities as points.
    
    Args:
        city_results: List of tuples (city_name, metrics_dict, census_data)
        output_dir: Directory to save output
    """
    if output_dir is None:
        output_dir = Path('city_comparison_outputs')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CREATING ALL-CITIES CENSUS VISUALIZATION")
    print(f"{'='*80}")
    
    # Prepare data for plotting
    plot_data = []
    for city_name, metrics, census_data in city_results:
        if census_data is None:
            continue
        
        pop_2024 = census_data.get('population_2024', np.nan)
        if pd.isna(pop_2024):
            continue
        
        plot_data.append({
            'city': city_name,
            'population': pop_2024,
            'total_listings': metrics.get('total_listings', np.nan),
            'median_price': metrics.get('median_price', np.nan),
            'median_occupancy': metrics.get('median_occupancy', np.nan) * 100 if 'median_occupancy' in metrics else np.nan,
            'median_roi': metrics.get('median_roi', np.nan),
        })
    
    if not plot_data:
        print("  ⚠️  No data to plot")
        return None
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Get unique cities for color palette
    n_cities = len(df_plot)
    if n_cities <= 10:
        colors = sns.color_palette("tab10", n_cities)
    elif n_cities <= 20:
        colors = sns.color_palette("tab20", n_cities)
    else:
        colors = sns.color_palette("husl", n_cities)
    
    city_colors = dict(zip(df_plot['city'], colors))
    
    # Plot 1: Population vs Market Size (listings)
    ax = axes[0, 0]
    for _, row in df_plot.iterrows():
        if pd.notna(row['total_listings']):
            ax.scatter(row['population'], row['total_listings'], 
                      s=200, alpha=0.7, color=city_colors[row['city']],
                      edgecolors='black', linewidth=1.5, zorder=5)
            # Add city label
            ax.annotate(row['city'], (row['population'], row['total_listings']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.8)
    ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Airbnb Listings', fontweight='bold', fontsize=12)
    ax.set_title('Population vs Market Size', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Population vs Median Price
    ax = axes[0, 1]
    for _, row in df_plot.iterrows():
        if pd.notna(row['median_price']):
            ax.scatter(row['population'], row['median_price'], 
                      s=200, alpha=0.7, color=city_colors[row['city']],
                      edgecolors='black', linewidth=1.5, zorder=5)
            # Add city label
            ax.annotate(row['city'], (row['population'], row['median_price']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.8)
    ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Median Nightly Price ($)', fontweight='bold', fontsize=12)
    ax.set_title('Population vs Median Price', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 3: Population vs Median Occupancy
    ax = axes[1, 0]
    for _, row in df_plot.iterrows():
        if pd.notna(row['median_occupancy']):
            ax.scatter(row['population'], row['median_occupancy'], 
                      s=200, alpha=0.7, color=city_colors[row['city']],
                      edgecolors='black', linewidth=1.5, zorder=5)
            # Add city label
            ax.annotate(row['city'], (row['population'], row['median_occupancy']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.8)
    ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Median Occupancy Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Population vs Occupancy Rate', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 4: Population vs ROI
    ax = axes[1, 1]
    for _, row in df_plot.iterrows():
        if pd.notna(row['median_roi']):
            ax.scatter(row['population'], row['median_roi'], 
                      s=200, alpha=0.7, color=city_colors[row['city']],
                      edgecolors='black', linewidth=1.5, zorder=5)
            # Add city label
            ax.annotate(row['city'], (row['population'], row['median_roi']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.8)
    ax.set_xlabel('Metropolitan Area Population (2024)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Median Cash-on-Cash ROI (%)', fontweight='bold', fontsize=12)
    ax.set_title('Population vs ROI', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig.suptitle('Census Data Exploration: All US Cities\n(Relationship Between Metropolitan Population and Airbnb Market Characteristics)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'all_cities_census_exploration.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Also save the data as CSV
    csv_file = output_dir / 'all_cities_census_data.csv'
    df_plot.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    return output_file


def main():
    """Main execution function"""
    use_detailed = '-all' in sys.argv
    
    # Check if user wants all cities (before filtering)
    wants_all_cities = '-all' in sys.argv[1:] and len([a for a in sys.argv[1:] if a != '-all']) == 0
    
    # Get city name (first non-flag argument)
    city_args = [arg for arg in sys.argv[1:] if arg != '-all']
    
    # List of all cities
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Boston', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Portland', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    
    if wants_all_cities or (city_args and (city_args[0] == '-all' or city_args[0].lower() == 'all')):
        # Filter out Paris (not US)
        city_folders = [c for c in all_cities if c != 'Paris']
        print(f"\n📊 BATCH MODE: Analyzing all {len(city_folders)} US cities (excluding Paris)")
    elif not city_args:
        print("Usage: python explore_census_data.py CityName [-all]")
        print("       python explore_census_data.py -all  # All cities")
        print("\nExample: python explore_census_data.py Austin -all")
        sys.exit(1)
    else:
        city_folders = [city_args[0]]
        print(f"\n🎯 SINGLE CITY MODE: Analyzing {city_folders[0]} only")
    
    if use_detailed:
        print(f"🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"🔍 MODE: SIMPLE ANALYSIS (19 variables)")
    
    # Load census data
    print(f"\n{'='*80}")
    print(f"LOADING CENSUS DATA")
    print(f"{'='*80}")
    try:
        census_df = load_census_data(base_dir='.')
    except Exception as e:
        print(f"❌ Error loading census data: {e}")
        sys.exit(1)
    
    results = []
    
    for city_name in city_folders:
        print(f"\n{'='*80}")
        print(f"Processing: {city_name}")
        print(f"{'='*80}")
        
        try:
            # Load city data
            df = load_city_data(city_name, base_dir='.', use_detailed=use_detailed)
            
            # Match to census
            census_match = match_city_to_census(city_name, census_df)
            
            # Explore relationships
            metrics = explore_census_relationships(city_name, df, census_match)
            
            if metrics:
                results.append((city_name, metrics, census_match))
        except Exception as e:
            print(f"  ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nSuccessfully analyzed {len(results)} cities")
    
    # If batch mode, create all-cities visualization
    if len(city_folders) > 1:
        print(f"\n{'='*80}")
        print(f"CREATING ALL-CITIES VISUALIZATION")
        print(f"{'='*80}")
        create_all_cities_census_visualization(results)
    
    return results


if __name__ == "__main__":
    main()

