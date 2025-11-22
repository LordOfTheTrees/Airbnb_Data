"""
Investigate why host_blocked_days can be negative
This happens when: unavailable_days < booked_days
Which means: (365 - availability_365) < estimated_occupancy_l365d
Or: availability_365 + estimated_occupancy_l365d > 365

This shouldn't be possible unless:
1. Data inconsistency between availability_365 and estimated_occupancy_l365d
2. Different time periods for the two metrics
3. Data quality issues
"""

import pandas as pd
from pathlib import Path

def investigate_negative_host_blocked(city_name='Austin', base_dir='.'):
    """Investigate negative host-blocked days"""
    city_path = Path(base_dir) / city_name
    
    # Load data
    listings_file = city_path / 'listings.csv.gz'
    if not listings_file.exists():
        listings_file = city_path / 'listings.csv'
    
    df = pd.read_csv(listings_file, compression='gzip' if str(listings_file).endswith('.gz') else None)
    
    # Filter hotel rooms
    if 'room_type' in df.columns:
        df = df[df['room_type'] != 'Hotel room'].copy()
    
    # Calculate metrics
    df['unavailable_days'] = 365 - df['availability_365']
    df['host_blocked_days'] = df['unavailable_days'] - df['estimated_occupancy_l365d']
    df['host_blocked_rate'] = df['host_blocked_days'] / 365
    
    # Find negative cases
    negative = df[df['host_blocked_days'] < 0].copy()
    
    print(f"\n{'='*80}")
    print(f"INVESTIGATING NEGATIVE HOST-BLOCKED DAYS: {city_name.upper()}")
    print(f"{'='*80}")
    print(f"\nTotal listings (excluding hotel rooms): {len(df):,}")
    print(f"Listings with negative host_blocked_days: {len(negative):,} ({len(negative)/len(df)*100:.2f}%)")
    
    if len(negative) > 0:
        print(f"\n{'='*80}")
        print("SAMPLE NEGATIVE CASES:")
        print(f"{'='*80}")
        print(negative[['availability_365', 'estimated_occupancy_l365d', 'unavailable_days', 'host_blocked_days']].head(20))
        
        print(f"\n{'='*80}")
        print("STATISTICS FOR NEGATIVE CASES:")
        print(f"{'='*80}")
        print(f"availability_365 range: {negative['availability_365'].min():.0f} - {negative['availability_365'].max():.0f}")
        print(f"estimated_occupancy_l365d range: {negative['estimated_occupancy_l365d'].min():.0f} - {negative['estimated_occupancy_l365d'].max():.0f}")
        print(f"unavailable_days range: {negative['unavailable_days'].min():.0f} - {negative['unavailable_days'].max():.0f}")
        print(f"host_blocked_days range: {negative['host_blocked_days'].min():.0f} - {negative['host_blocked_days'].max():.0f}")
        
        # Check if availability_365 + estimated_occupancy_l365d > 365
        negative['sum_check'] = negative['availability_365'] + negative['estimated_occupancy_l365d']
        print(f"\navailability_365 + estimated_occupancy_l365d:")
        print(f"  Min: {negative['sum_check'].min():.0f}")
        print(f"  Max: {negative['sum_check'].max():.0f}")
        print(f"  Mean: {negative['sum_check'].mean():.1f}")
        print(f"  Count > 365: {(negative['sum_check'] > 365).sum():,}")
        print(f"  Count > 400: {(negative['sum_check'] > 400).sum():,}")
        
        # Possible explanations
        print(f"\n{'='*80}")
        print("POSSIBLE EXPLANATIONS:")
        print(f"{'='*80}")
        print("1. Different time periods: availability_365 might be for a different 365-day period")
        print("   than estimated_occupancy_l365d")
        print("2. Data inconsistency: The two metrics may be calculated differently")
        print("3. Overlapping bookings: If a property has multiple bookings on the same day")
        print("4. Data quality: Missing or incorrect values")
        
        # Check room types
        if 'room_type' in negative.columns:
            print(f"\nRoom type distribution for negative cases:")
            print(negative['room_type'].value_counts())
    
    return negative

if __name__ == "__main__":
    investigate_negative_host_blocked('Austin')

