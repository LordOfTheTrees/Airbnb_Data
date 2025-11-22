"""Compare our dataset with Kaggle dataset for validation"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_our_chicago_data():
    """Load Chicago data from our dataset"""
    chicago_path = Path('Chicago')
    
    # Try listings.csv.gz first, then listings.csv
    if (chicago_path / 'listings.csv.gz').exists():
        df = pd.read_csv(chicago_path / 'listings.csv.gz', compression='gzip')
    elif (chicago_path / 'listings.csv').exists():
        df = pd.read_csv(chicago_path / 'listings.csv')
    else:
        raise FileNotFoundError("Chicago listings file not found")
    
    # Clean price
    if 'price' in df.columns:
        df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
        # Calculate log_price for comparison
        df['log_price'] = np.log(df['price_clean'].clip(lower=1))
    
    return df

def load_kaggle_chicago_data():
    """Load Chicago data from Kaggle dataset"""
    df = pd.read_csv('Kaggle/Airbnb_Data.csv', low_memory=False)
    
    # Filter to Chicago
    # Kaggle uses "Chicago" as city name
    chicago_data = df[df['city'] == 'Chicago'].copy()
    
    return chicago_data

def compare_datasets(our_df, kaggle_df):
    """Compare key metrics between datasets"""
    
    print("="*80)
    print("DATASET COMPARISON: CHICAGO")
    print("="*80)
    
    print(f"\n{'METRIC':<40} {'OUR DATASET':<20} {'KAGGLE':<20} {'DIFFERENCE':<15}")
    print("-"*95)
    
    # Basic counts
    print(f"{'Total listings':<40} {len(our_df):<20,} {len(kaggle_df):<20,} {len(our_df) - len(kaggle_df):<15,}")
    
    # Property type comparison
    print(f"\n{'PROPERTY TYPE DISTRIBUTION':<40}")
    print("-"*95)
    our_prop = our_df['property_type'].value_counts().head(10)
    kaggle_prop = kaggle_df['property_type'].value_counts().head(10)
    
    all_prop_types = set(our_prop.index) | set(kaggle_prop.index)
    for prop_type in sorted(all_prop_types)[:10]:
        our_count = our_prop.get(prop_type, 0)
        kaggle_count = kaggle_prop.get(prop_type, 0)
        our_pct = our_count / len(our_df) * 100
        kaggle_pct = kaggle_count / len(kaggle_df) * 100 if len(kaggle_df) > 0 else 0
        diff = our_pct - kaggle_pct
        print(f"  {prop_type:<38} {our_count:>6,} ({our_pct:>5.1f}%) {kaggle_count:>6,} ({kaggle_pct:>5.1f}%) {diff:>6.1f}%")
    
    # Room type comparison
    print(f"\n{'ROOM TYPE DISTRIBUTION':<40}")
    print("-"*95)
    our_room = our_df['room_type'].value_counts()
    kaggle_room = kaggle_df['room_type'].value_counts()
    
    all_room_types = set(our_room.index) | set(kaggle_room.index)
    for room_type in sorted(all_room_types):
        our_count = our_room.get(room_type, 0)
        kaggle_count = kaggle_room.get(room_type, 0)
        our_pct = our_count / len(our_df) * 100
        kaggle_pct = kaggle_count / len(kaggle_df) * 100 if len(kaggle_df) > 0 else 0
        diff = our_pct - kaggle_pct
        print(f"  {room_type:<38} {our_count:>6,} ({our_pct:>5.1f}%) {kaggle_count:>6,} ({kaggle_pct:>5.1f}%) {diff:>6.1f}%")
    
    # Numeric comparisons
    print(f"\n{'NUMERIC VARIABLES (mean)':<40}")
    print("-"*95)
    
    numeric_vars = ['accommodates', 'bedrooms', 'beds', 'number_of_reviews']
    for var in numeric_vars:
        if var in our_df.columns and var in kaggle_df.columns:
            our_mean = our_df[var].mean()
            kaggle_mean = kaggle_df[var].mean()
            diff = our_mean - kaggle_mean
            diff_pct = (diff / kaggle_mean * 100) if kaggle_mean != 0 else 0
            print(f"  {var:<38} {our_mean:>10.2f} {kaggle_mean:>10.2f} {diff:>10.2f} ({diff_pct:>5.1f}%)")
    
    # Price comparison (log_price for Kaggle, calculated log_price for ours)
    print(f"\n{'PRICE COMPARISON':<40}")
    print("-"*95)
    if 'log_price' in our_df.columns and 'log_price' in kaggle_df.columns:
        our_log_price = our_df['log_price'].dropna()
        kaggle_log_price = kaggle_df['log_price'].dropna()
        
        print(f"  {'Mean log_price':<38} {our_log_price.mean():>10.2f} {kaggle_log_price.mean():>10.2f} {our_log_price.mean() - kaggle_log_price.mean():>10.2f}")
        print(f"  {'Median log_price':<38} {our_log_price.median():>10.2f} {kaggle_log_price.median():>10.2f} {our_log_price.median() - kaggle_log_price.median():>10.2f}")
        print(f"  {'Std log_price':<38} {our_log_price.std():>10.2f} {kaggle_log_price.std():>10.2f} {our_log_price.std() - kaggle_log_price.std():>10.2f}")
        
        # Convert back to dollars for easier interpretation
        if 'price_clean' in our_df.columns:
            our_price = our_df['price_clean'].dropna()
            our_mean_price = our_price.mean()
            kaggle_mean_price = np.exp(kaggle_log_price.mean())
            print(f"  {'Mean price ($)':<38} ${our_mean_price:>9,.2f} ${kaggle_mean_price:>9,.2f} ${our_mean_price - kaggle_mean_price:>9,.2f}")
    
    # Review scores
    print(f"\n{'REVIEW SCORES':<40}")
    print("-"*95)
    if 'review_scores_rating' in our_df.columns and 'review_scores_rating' in kaggle_df.columns:
        our_reviews = our_df['review_scores_rating'].dropna()
        kaggle_reviews = kaggle_df['review_scores_rating'].dropna()
        
        print(f"  {'Mean rating':<38} {our_reviews.mean():>10.2f} {kaggle_reviews.mean():>10.2f} {our_reviews.mean() - kaggle_reviews.mean():>10.2f}")
        print(f"  {'Median rating':<38} {our_reviews.median():>10.2f} {kaggle_reviews.median():>10.2f} {our_reviews.median() - kaggle_reviews.median():>10.2f}")
        print(f"  {'Valid ratings':<38} {len(our_reviews):>10,} {len(kaggle_reviews):>10,} {len(our_reviews) - len(kaggle_reviews):>10,}")
    
    # ID overlap check
    print(f"\n{'ID OVERLAP ANALYSIS':<40}")
    print("-"*95)
    if 'id' in our_df.columns and 'id' in kaggle_df.columns:
        our_ids = set(our_df['id'].astype(str))
        kaggle_ids = set(kaggle_df['id'].astype(str))
        
        overlap = our_ids & kaggle_ids
        our_only = our_ids - kaggle_ids
        kaggle_only = kaggle_ids - our_ids
        
        print(f"  {'Our IDs':<38} {len(our_ids):>10,}")
        print(f"  {'Kaggle IDs':<38} {len(kaggle_ids):>10,}")
        print(f"  {'Overlapping IDs':<38} {len(overlap):>10,} ({len(overlap)/len(our_ids)*100:.1f}% of ours)")
        print(f"  {'Only in our dataset':<38} {len(our_only):>10,}")
        print(f"  {'Only in Kaggle':<38} {len(kaggle_only):>10,}")
        
        if len(overlap) > 0:
            print(f"\n  {'MATCHED LISTINGS COMPARISON (for overlapping IDs)':<40}")
            print("-"*95)
            matched_our = our_df[our_df['id'].astype(str).isin(overlap)]
            matched_kaggle = kaggle_df[kaggle_df['id'].astype(str).isin(overlap)]
            
            # Compare key fields for matched listings
            if 'log_price' in matched_our.columns and 'log_price' in matched_kaggle.columns:
                our_matched_price = matched_our['log_price'].dropna()
                kaggle_matched_price = matched_kaggle['log_price'].dropna()
                common_ids = set(matched_our['id'].astype(str)) & set(matched_kaggle['id'].astype(str))
                
                if len(common_ids) > 0:
                    # Direct comparison for matched IDs
                    matched_our_sorted = matched_our.set_index('id').loc[[int(x) for x in common_ids if x.isdigit()], 'log_price']
                    matched_kaggle_sorted = matched_kaggle.set_index('id').loc[[int(x) for x in common_ids if x.isdigit()], 'log_price']
                    
                    # Try to align
                    try:
                        price_diff = matched_our_sorted - matched_kaggle_sorted
                        print(f"  {'Price difference (mean)':<38} {price_diff.mean():>10.4f}")
                        print(f"  {'Price difference (std)':<38} {price_diff.std():>10.4f}")
                        print(f"  {'Exact price matches':<38} {(price_diff == 0).sum():>10,} ({(price_diff == 0).sum()/len(price_diff)*100:.1f}%)")
                    except:
                        print(f"  {'Could not align IDs for direct comparison':<38}")
    
    # Date range comparison (if available)
    print(f"\n{'DATE RANGES':<40}")
    print("-"*95)
    if 'host_since' in our_df.columns and 'host_since' in kaggle_df.columns:
        our_dates = pd.to_datetime(our_df['host_since'], errors='coerce').dropna()
        kaggle_dates = pd.to_datetime(kaggle_df['host_since'], errors='coerce').dropna()
        
        if len(our_dates) > 0 and len(kaggle_dates) > 0:
            print(f"  {'Our earliest host_since':<38} {our_dates.min()}")
            print(f"  {'Our latest host_since':<38} {our_dates.max()}")
            print(f"  {'Kaggle earliest host_since':<38} {kaggle_dates.min()}")
            print(f"  {'Kaggle latest host_since':<38} {kaggle_dates.max()}")
    
    if 'last_review' in our_df.columns and 'last_review' in kaggle_df.columns:
        our_reviews = pd.to_datetime(our_df['last_review'], errors='coerce').dropna()
        kaggle_reviews = pd.to_datetime(kaggle_df['last_review'], errors='coerce').dropna()
        
        if len(our_reviews) > 0 and len(kaggle_reviews) > 0:
            print(f"  {'Our earliest last_review':<38} {our_reviews.min()}")
            print(f"  {'Our latest last_review':<38} {our_reviews.max()}")
            print(f"  {'Kaggle earliest last_review':<38} {kaggle_reviews.min()}")
            print(f"  {'Kaggle latest last_review':<38} {kaggle_reviews.max()}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Overall assessment
    if 'id' in our_df.columns and 'id' in kaggle_df.columns:
        our_ids = set(our_df['id'].astype(str))
        kaggle_ids = set(kaggle_df['id'].astype(str))
        overlap = our_ids & kaggle_ids
        
        if len(overlap) > 0:
            overlap_pct = len(overlap) / len(our_ids) * 100
            print(f"\n{overlap_pct:.1f}% of our listings have matching IDs in Kaggle dataset")
            if overlap_pct > 50:
                print("  -> High overlap suggests same data source, different time periods or filters")
            elif overlap_pct > 10:
                print("  -> Moderate overlap suggests partial overlap or different sampling")
            else:
                print("  -> Low overlap suggests different data sources or different listings")
        else:
            print("\nWARNING: No ID overlap - datasets may be from different sources or time periods")
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print("\nLoading datasets...")
    
    try:
        our_df = load_our_chicago_data()
        print(f"Loaded our Chicago dataset: {len(our_df):,} listings")
    except Exception as e:
        print(f"ERROR loading our dataset: {e}")
        return
    
    try:
        kaggle_df = load_kaggle_chicago_data()
        print(f"Loaded Kaggle Chicago dataset: {len(kaggle_df):,} listings")
    except Exception as e:
        print(f"ERROR loading Kaggle dataset: {e}")
        return
    
    compare_datasets(our_df, kaggle_df)

if __name__ == "__main__":
    main()
