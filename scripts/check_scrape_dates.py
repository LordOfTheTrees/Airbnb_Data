"""Check when data was actually scraped/collected"""
import pandas as pd
from pathlib import Path

def check_scrape_dates(city_name='Chicago'):
    """Check last_scraped dates to see when data was collected"""
    city_path = Path(city_name)
    
    # Load data
    if (city_path / 'listings.csv.gz').exists():
        df = pd.read_csv(city_path / 'listings.csv.gz', compression='gzip', nrows=10000)  # Sample for speed
    elif (city_path / 'listings.csv').exists():
        df = pd.read_csv(city_path / 'listings.csv', nrows=10000)
    else:
        print(f"File not found for {city_name}")
        return None, None
    
    # Check if last_scraped exists
    if 'last_scraped' in df.columns:
        df['last_scraped'] = pd.to_datetime(df['last_scraped'], errors='coerce')
        valid = df['last_scraped'].dropna()
        
        if len(valid) == 0:
            print(f"\n{city_name}: last_scraped column exists but all values are missing")
            return None, None
        
        print(f"\n{city_name} - Data Collection Dates:")
        print(f"  Earliest scrape: {valid.min()}")
        print(f"  Latest scrape: {valid.max()}")
        print(f"  Date range: {(valid.max() - valid.min()).days} days")
        print(f"  Unique scrape dates: {valid.nunique()}")
        
        # Show distribution
        date_counts = valid.dt.date.value_counts().head(10)
        print(f"\n  Top scrape dates:")
        for date, count in date_counts.items():
            print(f"    {date}: {count:,} listings")
        
        return valid.min(), valid.max()
    else:
        print(f"\n{city_name}: WARNING - 'last_scraped' column not found!")
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'scraped' in c.lower() or 'time' in c.lower()]
        if date_cols:
            print(f"  Available date/timestamp columns: {date_cols}")
        return None, None

# Check multiple cities
print("="*80)
print("CHECKING DATA COLLECTION DATES (last_scraped)")
print("="*80)

cities = ['Chicago', 'Austin', 'New_York', 'Los_Angeles', 'San_Francisco', 'Seattle']
scrape_dates = {}

for city in cities:
    min_date, max_date = check_scrape_dates(city)
    if min_date and max_date:
        scrape_dates[city] = (min_date, max_date)

# Summary
if scrape_dates:
    print("\n" + "="*80)
    print("SUMMARY - Data Collection Date Ranges")
    print("="*80)
    
    all_min = min([d[0] for d in scrape_dates.values()])
    all_max = max([d[1] for d in scrape_dates.values()])
    
    print(f"\nOverall date range across all cities:")
    print(f"  Earliest: {all_min}")
    print(f"  Latest: {all_max}")
    print(f"  Total span: {(all_max - all_min).days} days ({(all_max - all_min).days / 365.25:.1f} years)")
    
    print(f"\nBy city:")
    for city, (min_date, max_date) in sorted(scrape_dates.items()):
        span_days = (max_date - min_date).days
        print(f"  {city:<20} {min_date.date()} to {max_date.date()} ({span_days} days)")
    
    # Check if all cities are from same time period
    date_ranges = [(d[0], d[1]) for d in scrape_dates.values()]
    if all(d[0].year == date_ranges[0][0].year and d[1].year == date_ranges[0][1].year for d in date_ranges):
        print(f"\n✓ All cities appear to be from same year: {date_ranges[0][0].year}")
        if all_max.year - all_min.year <= 1:
            print("  -> Prices likely from same time period, inflation adjustment NOT needed for cross-city comparison")
        else:
            print(f"  -> WARNING: {all_max.year - all_min.year} year span - may need inflation adjustment")
    else:
        print(f"\n⚠ Cities span multiple years - inflation adjustment may be needed")

