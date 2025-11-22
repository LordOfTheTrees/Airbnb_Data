"""
Analyze ROI metrics by property type and size segments within each city.

This script segments listings by:
- Property type (Entire home/apt, Private room, Shared room)
- Size bins (Studio, 1BR, 2BR, 3+BR) using bedrooms

For each segment within each city, calculates:
- Median cash_on_cash_roi
- Median cap_rate
- Median revenue_yield
- Median occupancy rate
- Count of listings (market size)

Output: Per-city ranking tables showing best segments by ROI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from city_level_analysis import apply_all_feature_engineering


def load_city_data(city_name, base_dir='.', use_detailed=False):
    """
    Load city data from CSV file.
    
    Args:
        city_name: Name of city folder
        base_dir: Base directory containing city folders
        use_detailed: If True, use detailed 79-variable dataset
        
    Returns:
        DataFrame with raw listing data
    """
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
    
    return df


def create_size_bins(bedrooms):
    """
    Create size bins from bedrooms.
    
    Args:
        bedrooms: Series of bedroom counts
        
    Returns:
        Series with size bin labels
    """
    bins = pd.Series(index=bedrooms.index, dtype='object')
    
    # Studio: 0 bedrooms
    bins[bedrooms == 0] = 'Studio'
    
    # 1BR: 1 bedroom
    bins[bedrooms == 1] = '1BR'
    
    # 2BR: 2 bedrooms
    bins[bedrooms == 2] = '2BR'
    
    # 3+BR: 3 or more bedrooms
    bins[bedrooms >= 3] = '3+BR'
    
    # Missing/unknown
    bins[bedrooms.isna()] = 'Unknown'
    
    return bins


def analyze_property_segments(df, city_name, output_dir=None):
    """
    Analyze ROI metrics by property type × size segments.
    
    Args:
        df: DataFrame with feature engineering applied (including ROI metrics)
        city_name: Name of city
        output_dir: Directory to save output files
        
    Returns:
        DataFrame with segment-level statistics
    """
    if output_dir is None:
        output_dir = Path(city_name) / 'analysis_output'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PROPERTY SEGMENT ANALYSIS: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Check required columns
    required_cols = ['property_type', 'room_type', 'bedrooms', 'cash_on_cash_roi', 
                     'cap_rate', 'revenue_yield', 'occupancy_rate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️  Missing required columns: {missing_cols}")
        print(f"     Skipping segment analysis for {city_name}")
        return None
    
    # Exclude hotel rooms from segment analysis
    # Rationale: Hotel rooms represent commercial operations, not residential rental investments
    df = df.copy()
    if 'room_type' in df.columns:
        n_hotel = (df['room_type'] == 'Hotel room').sum()
        if n_hotel > 0:
            df = df[df['room_type'] != 'Hotel room'].copy()
            print(f"  🏨 Excluded {n_hotel:,} hotel room listings from segment analysis")
    
    # Create size bins
    df['size_bin'] = create_size_bins(df['bedrooms'])
    
    # Create segment identifier: room_type × size_bin
    df['segment'] = df['room_type'].astype(str) + ' × ' + df['size_bin'].astype(str)
    
    # Filter to listings with valid ROI data
    valid_roi = df['cash_on_cash_roi'].notna()
    print(f"\n  Total listings: {len(df):,}")
    print(f"  Listings with ROI data: {valid_roi.sum():,} ({valid_roi.sum()/len(df)*100:.1f}%)")
    
    if valid_roi.sum() == 0:
        print(f"  ⚠️  No valid ROI data for {city_name}")
        return None
    
    # Calculate segment-level statistics
    segment_stats = []
    
    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        segment_valid = segment_data[valid_roi]
        
        if len(segment_valid) == 0:
            continue
        
        # Get room_type and size_bin from first row
        room_type = segment_data['room_type'].iloc[0]
        size_bin = segment_data['size_bin'].iloc[0]
        
        stats = {
            'room_type': room_type,
            'size_bin': size_bin,
            'segment': segment,
            'n_listings': len(segment_data),
            'n_with_roi': len(segment_valid),
            'median_cash_on_cash_roi': segment_valid['cash_on_cash_roi'].median(),
            'median_cap_rate': segment_valid['cap_rate'].median(),
            'median_revenue_yield': segment_valid['revenue_yield'].median(),
            'median_occupancy_rate': segment_valid['occupancy_rate'].median(),
            'median_est_annual_revenue': segment_valid['est_annual_revenue'].median() if 'est_annual_revenue' in segment_valid.columns else np.nan,
            'median_price_clean': segment_valid['price_clean'].median() if 'price_clean' in segment_valid.columns else np.nan,
            'pct_of_market': len(segment_data) / len(df) * 100,
        }
        
        segment_stats.append(stats)
    
    # Create DataFrame
    segments_df = pd.DataFrame(segment_stats)
    
    if len(segments_df) == 0:
        print(f"  ⚠️  No valid segments found for {city_name}")
        return None
    
    # Sort by median cash-on-cash ROI (descending)
    segments_df = segments_df.sort_values('median_cash_on_cash_roi', ascending=False)
    
    # Add ranking
    segments_df['roi_rank'] = range(1, len(segments_df) + 1)
    
    # Save to CSV
    output_file = output_dir / f'{city_name}_property_segments.csv'
    segments_df.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved segment analysis to {output_file}")
    
    # Print summary
    print(f"\n  Top 5 segments by cash-on-cash ROI:")
    print(f"  {'Rank':<6} {'Segment':<40} {'ROI %':<10} {'Cap Rate %':<12} {'Listings':<10}")
    print(f"  {'-'*80}")
    
    for idx, row in segments_df.head(5).iterrows():
        print(f"  {row['roi_rank']:<6} {row['segment']:<40} {row['median_cash_on_cash_roi']:>8.1f}% {row['median_cap_rate']:>10.2f}% {row['n_listings']:>8,}")
    
    return segments_df


def analyze_all_cities(city_folders, base_dir='.', use_detailed=False):
    """
    Analyze property segments for all cities.
    
    Args:
        city_folders: List of city folder names
        base_dir: Base directory
        use_detailed: If True, use detailed 79-variable datasets
    """
    print(f"\n{'='*80}")
    print(f"PROPERTY SEGMENT ANALYSIS - ALL CITIES")
    print(f"{'='*80}")
    
    all_segments = []
    
    for city_name in city_folders:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {city_name}")
            print(f"{'='*80}")
            
            # Load and process city data
            df = load_city_data(city_name, base_dir, use_detailed)
            
            # Apply feature engineering (including Zillow and ROI)
            df = apply_all_feature_engineering(df, city_name, include_zillow=True)
            
            # Analyze segments
            segments_df = analyze_property_segments(df, city_name)
            
            if segments_df is not None:
                segments_df['city'] = city_name
                all_segments.append(segments_df)
            
        except Exception as e:
            print(f"  ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all cities
    if all_segments:
        combined_df = pd.concat(all_segments, ignore_index=True)
        
        # Save combined results
        output_file = Path(base_dir) / 'city_comparison_outputs' / 'property_segments_all_cities.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"COMBINED RESULTS")
        print(f"{'='*80}")
        print(f"  ✓ Saved combined results to {output_file}")
        print(f"  Total segments analyzed: {len(combined_df)}")
        print(f"  Cities analyzed: {len(all_segments)}")
    
    return all_segments


if __name__ == "__main__":
    """
    Run property segment analysis.
    
    Usage:
        python analyze_property_segments.py                    # All cities, simple datasets
        python analyze_property_segments.py -all              # All cities, detailed datasets
        python analyze_property_segments.py Austin            # Just Austin, simple dataset
        python analyze_property_segments.py Austin -all        # Just Austin, detailed dataset
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    
    # Check if a specific city was requested
    city_args = [arg for arg in sys.argv[1:] if arg != '-all']
    single_city = city_args[0] if city_args else None
    
    # City list (same as city_level_analysis.py)
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    
    # Determine which cities to analyze
    if single_city:
        city_folders = [c for c in all_cities if c.lower() == single_city.lower()]
        
        if not city_folders:
            print(f"\n❌ ERROR: City '{single_city}' not found in city list!")
            print(f"\nAvailable cities:")
            for city in all_cities:
                print(f"  - {city}")
            sys.exit(1)
        
        print(f"\n🎯 SINGLE CITY MODE: Analyzing {city_folders[0]} only")
    else:
        city_folders = all_cities
        print(f"\n📊 BATCH MODE: Analyzing all {len(city_folders)} cities")
    
    if use_detailed:
        print(f"🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"🔍 MODE: SIMPLE ANALYSIS (19 variables)")
    
    # Run analysis
    results = analyze_all_cities(city_folders, base_dir='.', use_detailed=use_detailed)
    
    print(f"\n{'='*80}")
    print(f"ALL DONE! Check each city's 'analysis_output' folder for results.")
    print(f"{'='*80}")

