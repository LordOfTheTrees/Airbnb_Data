"""
Inflation Adjustment Utility
Adjusts prices to a common year using Consumer Price Index (CPI) data
"""

import pandas as pd
import numpy as np
from datetime import datetime

# CPI-U (Consumer Price Index for All Urban Consumers) - Base year 1982-84 = 100
# Source: U.S. Bureau of Labor Statistics
# Annual averages
CPI_DATA = {
    2008: 215.303,
    2009: 214.537,
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.127,
    2024: 313.548,  # Estimated (through Nov 2024)
    2025: 320.000,  # Projected estimate
}

def get_cpi(year):
    """Get CPI for a given year"""
    if year in CPI_DATA:
        return CPI_DATA[year]
    elif year < min(CPI_DATA.keys()):
        # Extrapolate backwards (rough estimate)
        return CPI_DATA[min(CPI_DATA.keys())] * (0.97 ** (min(CPI_DATA.keys()) - year))
    elif year > max(CPI_DATA.keys()):
        # Extrapolate forwards (rough estimate, 3% inflation)
        return CPI_DATA[max(CPI_DATA.keys())] * (1.03 ** (year - max(CPI_DATA.keys())))
    else:
        # Interpolate
        years = sorted(CPI_DATA.keys())
        for i, y in enumerate(years):
            if y > year:
                if i == 0:
                    return CPI_DATA[y]
                y1, y2 = years[i-1], y
                cpi1, cpi2 = CPI_DATA[y1], CPI_DATA[y2]
                # Linear interpolation
                return cpi1 + (cpi2 - cpi1) * (year - y1) / (y2 - y1)
        return CPI_DATA[max(years)]

def adjust_price_for_inflation(price, from_year, to_year=2024):
    """
    Adjust a price from one year to another using CPI
    
    Args:
        price: Price in dollars
        from_year: Year the price is from
        to_year: Year to adjust to (default: 2024)
    
    Returns:
        Adjusted price in to_year dollars
    """
    if pd.isna(price) or pd.isna(from_year):
        return np.nan
    
    from_year = int(from_year)
    to_year = int(to_year)
    
    if from_year == to_year:
        return price
    
    cpi_from = get_cpi(from_year)
    cpi_to = get_cpi(to_year)
    
    if cpi_from == 0:
        return np.nan
    
    adjusted_price = price * (cpi_to / cpi_from)
    return adjusted_price

def add_inflation_adjusted_prices(df, price_col='price_clean', 
                                   date_col='last_review', 
                                   base_year=2024,
                                   create_adjusted=True):
    """
    Add inflation-adjusted prices to dataframe
    
    Args:
        df: DataFrame with price and date columns
        price_col: Name of price column
        date_col: Name of date column to extract year from (e.g., 'last_review', 'host_since')
        base_year: Year to adjust all prices to (default: 2024)
        create_adjusted: If True, create new column with adjusted prices
    
    Returns:
        DataFrame with inflation-adjusted price column
    """
    df = df.copy()
    
    if price_col not in df.columns:
        print(f"WARNING: {price_col} not found in dataframe")
        return df
    
    if date_col not in df.columns:
        print(f"WARNING: {date_col} not found in dataframe - cannot determine year")
        return df
    
    # Extract year from date column
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    df['_price_year'] = df[date_col].dt.year
    
    # Adjust prices
    if create_adjusted:
        adjusted_col = f'{price_col}_adjusted_{base_year}'
        df[adjusted_col] = df.apply(
            lambda row: adjust_price_for_inflation(
                row[price_col], 
                row['_price_year'], 
                base_year
            ) if pd.notna(row[price_col]) and pd.notna(row['_price_year']) else np.nan,
            axis=1
        )
        
        n_adjusted = df[adjusted_col].notna().sum()
        print(f"Created {adjusted_col}: {n_adjusted:,} adjusted prices")
        print(f"  Base year: {base_year}")
        print(f"  Source year range: {df['_price_year'].min():.0f} - {df['_price_year'].max():.0f}")
    
    # Clean up temporary column
    if '_price_year' in df.columns:
        df = df.drop('_price_year', axis=1)
    
    return df

def compare_inflation_impact(df, price_col='price_clean', date_col='last_review'):
    """
    Show impact of inflation adjustment on prices
    
    Args:
        df: DataFrame with price and date columns
        price_col: Name of price column
        date_col: Name of date column
    """
    if price_col not in df.columns or date_col not in df.columns:
        print("Required columns not found")
        return
    
    # Extract year
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    df['_year'] = df[date_col].dt.year
    valid = df[[price_col, '_year']].dropna()
    
    if len(valid) == 0:
        print("No valid price/date data")
        return
    
    print("\n" + "="*80)
    print("INFLATION IMPACT ANALYSIS")
    print("="*80)
    
    print(f"\nPrice data by year:")
    year_stats = valid.groupby('_year')[price_col].agg(['count', 'mean', 'median'])
    print(year_stats.to_string())
    
    # Adjust to 2024
    valid['price_2024'] = valid.apply(
        lambda row: adjust_price_for_inflation(row[price_col], row['_year'], 2024),
        axis=1
    )
    
    print(f"\nOriginal prices (by year):")
    print(f"  Mean: ${valid[price_col].mean():,.2f}")
    print(f"  Median: ${valid[price_col].median():,.2f}")
    
    print(f"\nAdjusted to 2024 dollars:")
    print(f"  Mean: ${valid['price_2024'].mean():,.2f}")
    print(f"  Median: ${valid['price_2024'].median():,.2f}")
    print(f"  Difference: ${valid['price_2024'].mean() - valid[price_col].mean():,.2f}")
    print(f"  % Change: {(valid['price_2024'].mean() / valid[price_col].mean() - 1) * 100:.1f}%")
    
    # Show adjustment factors by year
    print(f"\nInflation adjustment factors (to 2024):")
    years = sorted(valid['_year'].unique())
    for year in years:
        cpi_year = get_cpi(year)
        cpi_2024 = get_cpi(2024)
        factor = cpi_2024 / cpi_year
        print(f"  {year}: {factor:.3f}x (${100:.2f} in {year} = ${100 * factor:.2f} in 2024)")

if __name__ == "__main__":
    # Test the inflation adjustment
    print("Testing inflation adjustment...")
    
    # Example: $100 in 2017 vs 2024
    price_2017 = 100
    price_2024 = adjust_price_for_inflation(price_2017, 2017, 2024)
    print(f"\n$100 in 2017 = ${price_2024:.2f} in 2024")
    
    # Show CPI values
    print("\nCPI values (1982-84 = 100):")
    for year in sorted(CPI_DATA.keys()):
        print(f"  {year}: {CPI_DATA[year]:.3f}")
    
    # Show adjustment factors to 2024
    print("\nAdjustment factors to 2024:")
    for year in [2017, 2020, 2022, 2023]:
        factor = get_cpi(2024) / get_cpi(year)
        print(f"  {year} → 2024: {factor:.3f}x")

