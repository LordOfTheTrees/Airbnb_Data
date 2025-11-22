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
    
    # Convert date columns to datetime and find the latest non-null date for each row
    latest_prices = []
    latest_dates = []
    
    for idx, row in df.iterrows():
        # Get all date values for this row
        date_values = []
        for col in date_cols:
            try:
                date_val = pd.to_datetime(col)
                price_val = row[col]
                if pd.notna(price_val) and price_val != '':
                    date_values.append((date_val, price_val))
            except:
                continue
        
        if date_values:
            # Sort by date and get the latest
            date_values.sort(key=lambda x: x[0])
            latest_date, latest_price = date_values[-1]
            latest_prices.append(latest_price)
            latest_dates.append(latest_date)
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
    
    # Extract latest values
    latest_rents = []
    latest_dates = []
    
    for idx, row in df.iterrows():
        date_values = []
        for col in date_cols:
            try:
                date_val = pd.to_datetime(col)
                rent_val = row[col]
                if pd.notna(rent_val) and rent_val != '':
                    date_values.append((date_val, rent_val))
            except:
                continue
        
        if date_values:
            date_values.sort(key=lambda x: x[0])
            latest_date, latest_rent = date_values[-1]
            latest_rents.append(latest_rent)
            latest_dates.append(latest_date)
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
    
    # Extract latest values
    latest_payments = []
    latest_dates = []
    
    for idx, row in df.iterrows():
        date_values = []
        for col in date_cols:
            try:
                date_val = pd.to_datetime(col)
                payment_val = row[col]
                if pd.notna(payment_val) and payment_val != '':
                    date_values.append((date_val, payment_val))
            except:
                continue
        
        if date_values:
            date_values.sort(key=lambda x: x[0])
            latest_date, latest_payment = date_values[-1]
            latest_payments.append(latest_payment)
            latest_dates.append(latest_date)
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

