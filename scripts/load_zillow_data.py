"""
Load and process Zillow data files.

This module provides functions to load Zillow ZHVI (home values), ZORI (rental index),
and monthly payment data, extracting the most recent month for each metro area.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_zillow_zhvi(zillow_dir='Zillow'):
    """
    Load Zillow Home Value Index (ZHVI) data and extract most recent month.
    
    Args:
        zillow_dir: Directory containing Zillow CSV files
        
    Returns:
        DataFrame with columns: RegionID, RegionName, StateName, zhvi_price, zhvi_date
    """
    file_path = Path(zillow_dir) / 'Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"ZHVI file not found: {file_path}")
    
    print(f"Loading ZHVI data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Find date columns (all columns except metadata columns)
    metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    date_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Convert date columns to datetime (vectorized)
    date_cols_parsed = []
    for col in date_cols:
        try:
            pd.to_datetime(col)  # Test if it's a valid date
            date_cols_parsed.append(col)
        except:
            continue
    
    if not date_cols_parsed:
        raise ValueError("No valid date columns found in ZHVI file")
    
    # Convert date column names to datetime for sorting
    date_cols_sorted = sorted(date_cols_parsed, key=lambda x: pd.to_datetime(x))
    
    # Extract date columns as numeric values
    date_data = df[date_cols_sorted].copy()
    
    # Find the last non-null value for each row (fully vectorized)
    # Use apply to find last non-null column index for each row
    def get_last_non_null(row):
        non_null = row[row.notna()]
        if len(non_null) > 0:
            return non_null.index[-1]
        return None
    
    last_col_indices = date_data.apply(get_last_non_null, axis=1)
    
    # Extract values and dates
    latest_prices = []
    latest_dates = []
    
    for idx, col_name in enumerate(last_col_indices):
        if col_name is not None:
            latest_prices.append(date_data.loc[df.index[idx], col_name])
            latest_dates.append(pd.to_datetime(col_name))
        else:
            latest_prices.append(np.nan)
            latest_dates.append(pd.NaT)
    
    # Create result DataFrame
    result = df[metadata_cols].copy()
    result['zhvi_price'] = latest_prices
    result['zhvi_date'] = latest_dates
    
    # Filter to only metros (RegionType == 'msa')
    result = result[result['RegionType'] == 'msa'].copy()
    
    # Convert price to numeric
    result['zhvi_price'] = pd.to_numeric(result['zhvi_price'], errors='coerce')
    
    print(f"  Loaded {len(result):,} metros")
    print(f"  Latest date range: {result['zhvi_date'].min()} to {result['zhvi_date'].max()}")
    print(f"  Price range: ${result['zhvi_price'].min():,.0f} to ${result['zhvi_price'].max():,.0f}")
    
    return result


def load_zillow_zori(zillow_dir='Zillow'):
    """
    Load Zillow Observed Rent Index (ZORI) data and extract most recent month.
    
    Args:
        zillow_dir: Directory containing Zillow CSV files
        
    Returns:
        DataFrame with columns: RegionID, RegionName, StateName, zori_rent, zori_date
    """
    file_path = Path(zillow_dir) / 'Metro_zori_uc_sfrcondomfr_sm_month.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"ZORI file not found: {file_path}")
    
    print(f"Loading ZORI data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Find date columns
    metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    date_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Convert date columns to datetime (vectorized)
    date_cols_parsed = []
    for col in date_cols:
        try:
            pd.to_datetime(col)  # Test if it's a valid date
            date_cols_parsed.append(col)
        except:
            continue
    
    if not date_cols_parsed:
        raise ValueError("No valid date columns found in ZORI file")
    
    # Convert date column names to datetime for sorting
    date_cols_sorted = sorted(date_cols_parsed, key=lambda x: pd.to_datetime(x))
    
    # Extract date columns as numeric values
    date_data = df[date_cols_sorted].copy()
    
    # Find the last non-null value for each row (fully vectorized)
    def get_last_non_null(row):
        non_null = row[row.notna()]
        if len(non_null) > 0:
            return non_null.index[-1]
        return None
    
    last_col_indices = date_data.apply(get_last_non_null, axis=1)
    
    # Extract values and dates
    latest_rents = []
    latest_dates = []
    
    for idx, col_name in enumerate(last_col_indices):
        if col_name is not None:
            latest_rents.append(date_data.loc[df.index[idx], col_name])
            latest_dates.append(pd.to_datetime(col_name))
        else:
            latest_rents.append(np.nan)
            latest_dates.append(pd.NaT)
    
    result = df[metadata_cols].copy()
    result['zori_rent'] = latest_rents
    result['zori_date'] = latest_dates
    
    # Filter to only metros
    result = result[result['RegionType'] == 'msa'].copy()
    
    # Convert to numeric
    result['zori_rent'] = pd.to_numeric(result['zori_rent'], errors='coerce')
    
    print(f"  Loaded {len(result):,} metros")
    print(f"  Latest date range: {result['zori_date'].min()} to {result['zori_date'].max()}")
    print(f"  Rent range: ${result['zori_rent'].min():,.0f} to ${result['zori_rent'].max():,.0f}")
    
    return result


def load_zillow_monthly_payment(zillow_dir='Zillow'):
    """
    Load Zillow monthly payment data (20% down, middle tier) and extract most recent month.
    
    Args:
        zillow_dir: Directory containing Zillow CSV files
        
    Returns:
        DataFrame with columns: RegionID, RegionName, StateName, monthly_payment, payment_date
    """
    file_path = Path(zillow_dir) / 'Metro_total_monthly_payment_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Monthly payment file not found: {file_path}")
    
    print(f"Loading monthly payment data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Find date columns
    metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    date_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Convert date columns to datetime (vectorized)
    date_cols_parsed = []
    for col in date_cols:
        try:
            pd.to_datetime(col)  # Test if it's a valid date
            date_cols_parsed.append(col)
        except:
            continue
    
    if not date_cols_parsed:
        raise ValueError("No valid date columns found in monthly payment file")
    
    # Convert date column names to datetime for sorting
    date_cols_sorted = sorted(date_cols_parsed, key=lambda x: pd.to_datetime(x))
    
    # Extract date columns as numeric values
    date_data = df[date_cols_sorted].copy()
    
    # Find the last non-null value for each row (fully vectorized)
    def get_last_non_null(row):
        non_null = row[row.notna()]
        if len(non_null) > 0:
            return non_null.index[-1]
        return None
    
    last_col_indices = date_data.apply(get_last_non_null, axis=1)
    
    # Extract values and dates
    latest_payments = []
    latest_dates = []
    
    for idx, col_name in enumerate(last_col_indices):
        if col_name is not None:
            latest_payments.append(date_data.loc[df.index[idx], col_name])
            latest_dates.append(pd.to_datetime(col_name))
        else:
            latest_payments.append(np.nan)
            latest_dates.append(pd.NaT)
    
    result = df[metadata_cols].copy()
    result['monthly_payment'] = latest_payments
    result['payment_date'] = latest_dates
    
    # Filter to only metros
    result = result[result['RegionType'] == 'msa'].copy()
    
    # Convert to numeric
    result['monthly_payment'] = pd.to_numeric(result['monthly_payment'], errors='coerce')
    
    print(f"  Loaded {len(result):,} metros")
    print(f"  Latest date range: {result['payment_date'].min()} to {result['payment_date'].max()}")
    print(f"  Payment range: ${result['monthly_payment'].min():,.0f} to ${result['monthly_payment'].max():,.0f}")
    
    return result


def load_all_zillow_data(zillow_dir='Zillow'):
    """
    Load all Zillow data and merge into a single DataFrame.
    
    Args:
        zillow_dir: Directory containing Zillow CSV files
        
    Returns:
        DataFrame with columns: RegionID, RegionName, StateName, zhvi_price, zori_rent, monthly_payment
    """
    print("=" * 80)
    print("LOADING ZILLOW DATA")
    print("=" * 80)
    
    # Load each dataset
    zhvi_df = load_zillow_zhvi(zillow_dir)
    zori_df = load_zillow_zori(zillow_dir)
    payment_df = load_zillow_monthly_payment(zillow_dir)
    
    # Merge on RegionID
    result = zhvi_df[['RegionID', 'RegionName', 'StateName', 'zhvi_price']].copy()
    
    result = result.merge(
        zori_df[['RegionID', 'zori_rent']],
        on='RegionID',
        how='left'
    )
    
    result = result.merge(
        payment_df[['RegionID', 'monthly_payment']],
        on='RegionID',
        how='left'
    )
    
    print("\n" + "=" * 80)
    print(f"MERGED DATA: {len(result):,} metros")
    print(f"  Metros with ZHVI: {result['zhvi_price'].notna().sum():,}")
    print(f"  Metros with ZORI: {result['zori_rent'].notna().sum():,}")
    print(f"  Metros with monthly payment: {result['monthly_payment'].notna().sum():,}")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # Test loading
    zillow_data = load_all_zillow_data()
    print("\nSample data:")
    print(zillow_data.head(10))

