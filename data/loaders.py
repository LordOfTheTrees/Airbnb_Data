"""
Data loading functions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from data.feature_engineering import apply_all_feature_engineering

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


def discover_city_folders(base_dir='.'):
    """
    Automatically discover city folders by looking for directories that contain
    listings.csv or listings.csv.gz files.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of city folder names (sorted)
    """
    base_path = Path(base_dir)
    city_folders = []
    
    # Directories to exclude (not cities)
    exclude_dirs = {
        '__pycache__', 'Census', 'Kaggle', 'Zillow', 'old_scripts',
        'city_comparison_outputs', '.git', 'portfolio_outputs',
        'analysis', 'data', 'visualization', 'utils', 'scripts'
    }
    
    # Look for directories that contain listings files
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in exclude_dirs:
            # Check if it contains listings.csv or listings.csv.gz
            if (item / 'listings.csv').exists() or (item / 'listings.csv.gz').exists():
                city_folders.append(item.name)
    
    return sorted(city_folders)
