"""
City-Level Analysis Script - ENHANCED WITH FEATURE ENGINEERING
Generates comprehensive statistics and correlation matrices for each city individually
NOW INCLUDES: Log transforms, within-city standardization, and revenue proxies

Run this from your main Airbnb_Data directory

⚠️ IMPORTANT: The detailed file is named listings.csv.gz (not listings_csv.gz)

Usage:
    python city_level_analysis.py                    # All cities, simple (19 vars)
    python city_level_analysis.py -all               # All cities, detailed (79 vars)
    python city_level_analysis.py Austin             # Just Austin, simple
    python city_level_analysis.py Austin -all        # Just Austin, detailed
    python city_level_analysis.py -all Chicago       # Order doesn't matter
    
Single City Mode (for debugging/development):
    python city_level_analysis.py Boston            # Fast testing on one city
    python city_level_analysis.py Boston -all       # Full analysis on one city
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_TEXT_LENGTH = 100  # Maximum characters for text fields in variable summary
TOP_CORRELATIONS_N = 25  # Number of top correlations to save and display
# ============================================================================

# ============================================================================
# FEATURE ENGINEERING - PRIORITY 1: LOG TRANSFORMATIONS
# ============================================================================

def add_log_transforms(df):
    """
    Priority 1: Add log transformations for key economic variables
    
    WHY: 
    - Investors think in percentages (returns), not absolute dollars
    - Price distributions are typically right-skewed
    - Log differences = percentage changes
    - Enables elasticity interpretation in correlations
    
    Transforms:
    - log_price: Natural log of price
    - log_price_per_accommodates: Natural log of price per guest capacity
    - log_reviews: Natural log of (number_of_reviews + 1) to handle zeros
    - log_accommodates: Natural log of guest capacity
    - log_beds: Natural log of (beds + 1) to handle zeros
    
    Args:
        df: DataFrame with raw listing data
    
    Returns:
        DataFrame with added log-transformed columns
    """
    print(f"\n  🔧 Priority 1: Adding log transformations...")
    
    df = df.copy()
    
    # Price transformations
    if 'price_clean' in df.columns:
        # Filter out zeros and negatives for log
        valid_price = df['price_clean'] > 0
        df.loc[valid_price, 'log_price'] = np.log(df.loc[valid_price, 'price_clean'])
        print(f"     ✓ Created log_price ({valid_price.sum():,} valid values)")
        
        # Price per accommodates (size-adjusted pricing)
        if 'accommodates' in df.columns:
            df['price_per_accommodates'] = df['price_clean'] / df['accommodates']
            valid_ppa = df['price_per_accommodates'] > 0
            df.loc[valid_ppa, 'log_price_per_accommodates'] = np.log(
                df.loc[valid_ppa, 'price_per_accommodates']
            )
            print(f"     ✓ Created log_price_per_accommodates ({valid_ppa.sum():,} valid values)")
    
    # Review transformations (use log1p to handle zeros gracefully)
    if 'number_of_reviews' in df.columns:
        df['log_reviews'] = np.log1p(df['number_of_reviews'])  # log(x + 1)
        print(f"     ✓ Created log_reviews (using log1p to handle zeros)")
    
    # Capacity transformations
    if 'accommodates' in df.columns:
        valid_acc = df['accommodates'] > 0
        df.loc[valid_acc, 'log_accommodates'] = np.log(df.loc[valid_acc, 'accommodates'])
        print(f"     ✓ Created log_accommodates ({valid_acc.sum():,} valid values)")
    
    if 'beds' in df.columns:
        df['log_beds'] = np.log1p(df['beds'])  # log(x + 1) for zeros
        print(f"     ✓ Created log_beds (using log1p to handle zeros)")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - PRIORITY 2: WITHIN-CITY STANDARDIZATION
# ============================================================================

def add_within_city_metrics(df, city_name):
    """
    Priority 2: Add within-city standardized metrics for cross-market comparison
    
    WHY:
    - A $200/night listing means different things in NYC vs Memphis
    - Investors care about relative positioning: "top 10% in this market"
    - Z-scores show how many standard deviations above/below average
    - Percentiles directly answer "better than X% of competition"
    
    Metrics:
    - price_zscore: Z-score of price within city (std devs from mean)
    - price_percentile: Percentile rank within city (0-1 scale)
    - log_price_zscore: Z-score of log price (for relative % differences)
    - reviews_zscore: Z-score of number of reviews
    - reviews_percentile: Percentile rank of reviews
    
    Args:
        df: DataFrame with listing data (should have price_clean, etc.)
        city_name: Name of the city (for display purposes)
    
    Returns:
        DataFrame with added standardized metrics
    """
    print(f"\n  🔧 Priority 2: Adding within-city standardization for {city_name}...")
    
    df = df.copy()
    
    # Price standardization
    if 'price_clean' in df.columns:
        # Z-score: how many standard deviations from mean?
        price_mean = df['price_clean'].mean()
        price_std = df['price_clean'].std()
        
        if price_std > 0:  # Avoid division by zero
            df['price_zscore'] = (df['price_clean'] - price_mean) / price_std
            print(f"     ✓ Created price_zscore (mean=${price_mean:.2f}, std=${price_std:.2f})")
        
        # Percentile rank: better than X% of listings
        df['price_percentile'] = df['price_clean'].rank(pct=True)
        print(f"     ✓ Created price_percentile (0=cheapest, 1=most expensive)")
    
    # Log price standardization (for relative percentage positioning)
    if 'log_price' in df.columns:
        log_price_mean = df['log_price'].mean()
        log_price_std = df['log_price'].std()
        
        if log_price_std > 0:
            df['log_price_zscore'] = (df['log_price'] - log_price_mean) / log_price_std
            print(f"     ✓ Created log_price_zscore (for relative % differences)")
    
    # Review count standardization (popularity/visibility metric)
    if 'number_of_reviews' in df.columns:
        reviews_mean = df['number_of_reviews'].mean()
        reviews_std = df['number_of_reviews'].std()
        
        if reviews_std > 0:
            df['reviews_zscore'] = (df['number_of_reviews'] - reviews_mean) / reviews_std
            print(f"     ✓ Created reviews_zscore (mean={reviews_mean:.1f}, std={reviews_std:.1f})")
        
        df['reviews_percentile'] = df['number_of_reviews'].rank(pct=True)
        print(f"     ✓ Created reviews_percentile (visibility ranking)")
    
    # Reviews per month standardization (activity velocity)
    if 'reviews_per_month' in df.columns:
        rpm_mean = df['reviews_per_month'].mean()
        rpm_std = df['reviews_per_month'].std()
        
        if rpm_std > 0:
            df['reviews_per_month_zscore'] = (df['reviews_per_month'] - rpm_mean) / rpm_std
            print(f"     ✓ Created reviews_per_month_zscore")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - PRIORITY 3: REVENUE PROXIES
# ============================================================================

def add_revenue_proxies(df):
    """
    Priority 3: Create revenue and ROI proxy metrics
    
    WHY:
    - Investors care about RETURNS, not just prices
    - Revenue = Price × Occupancy × Days
    - Size-adjusted metrics enable fair comparison
    - These metrics directly answer "which property makes more money?"
    
    OCCUPANCY METRIC RATIONALE:
    We use calendar-based occupancy (calendar_unavailable_proxy) as PRIMARY because:
    
    1. OPPORTUNITY COST ARGUMENT: When hosts block days, they implicitly value those days
       at least as much as the revenue they forego. This represents potential revenue
       under different management strategies, which is relevant for investment decisions.
    
    2. MARKET CAPACITY ARGUMENT: Calendar unavailability (booked + blocked) reflects
       total market demand. Whether unavailable due to booking or blocking, the property
       was in demand, showing its market capacity.
    
    3. DISTRIBUTION COMPLETENESS: estimated_occupancy_l365d is capped at 255 days (69.9%),
       creating artificial ceiling and downward bias. Calendar proxy provides full range.
    
    4. INVESTMENT VALUATION: For ROI analysis, we want maximum potential revenue.
       Host-blocked days represent revenue that could be captured with optimal management.
    
    5. BEHAVIORAL SIGNAL: High host-blocking rates signal high property value - hosts
       are willing to forego rental income, indicating desirability.
    
    6. COMPARABILITY: Calendar proxy enables fair comparison across all properties,
       distinguishing truly high-occupancy properties from those artificially capped.
    
    We keep estimated_occupancy_l365d as SECONDARY (occupancy_rate_booked) for comparison
    and to understand the difference between booked days and total unavailability.
    
    Metrics:
    - occupancy_rate: PRIMARY - Calendar-based (booked + host-blocked days) / 365
    - occupancy_rate_booked: SECONDARY - Actual booked days from Airbnb / 365
    - est_annual_revenue: PRIMARY - Revenue using calendar-based occupancy
    - est_annual_revenue_booked: SECONDARY - Revenue using actual booked days
    - revenue_per_accommodates: Revenue per guest capacity (unit economics)
    - revenue_per_bedroom: Revenue per bedroom (if bedrooms available)
    
    Args:
        df: DataFrame with price_clean, availability_365, and optionally estimated_occupancy_l365d
    
    Returns:
        DataFrame with added revenue proxy columns
    """
    print(f"\n  🔧 Priority 3: Adding revenue proxy metrics...")
    print(f"     Using calendar-based occupancy as PRIMARY (includes host-blocked days)")
    print(f"     Rationale: Captures full market capacity and potential revenue")
    
    df = df.copy()
    
    # PRIMARY: Calendar-based occupancy (includes booked + host-blocked days)
    if 'availability_365' in df.columns:
        df['occupancy_rate'] = (365 - df['availability_365']) / 365
        valid_occ = df['occupancy_rate'].notna()
        print(f"     ✓ Created occupancy_rate (PRIMARY) from calendar availability ({valid_occ.sum():,} valid values)")
        print(f"        Includes booked days + host-blocked days (full market capacity)")
    else:
        print(f"     ⚠️  availability_365 not found - cannot create primary occupancy_rate")
    
    # SECONDARY: Actual booked days from Airbnb (for comparison)
    if 'estimated_occupancy_l365d' in df.columns:
        df['occupancy_rate_booked'] = df['estimated_occupancy_l365d'] / 365
        valid_booked = df['occupancy_rate_booked'].notna()
        
        # Flag capped listings (255 days = 69.9% max due to 8-bit integer cap)
        df['occupancy_is_capped'] = (df['estimated_occupancy_l365d'] == 255).astype(int)
        n_capped = df['occupancy_is_capped'].sum()
        
        print(f"     ✓ Created occupancy_rate_booked (SECONDARY) from estimated_occupancy_l365d ({valid_booked.sum():,} valid values)")
        print(f"        {n_capped:,} listings capped at 255 days ({n_capped/valid_booked.sum()*100:.1f}%)")
        print(f"        Only includes actual booked days (excludes host-blocked days)")
    else:
        print(f"     ⚠️  estimated_occupancy_l365d not found - cannot create secondary occupancy_rate_booked")
    
    # PRIMARY: Estimated annual revenue using calendar-based occupancy
    if 'price_clean' in df.columns and 'availability_365' in df.columns:
        df['est_annual_revenue'] = df['price_clean'] * (365 - df['availability_365'])
        valid_rev = df['est_annual_revenue'].notna()
        print(f"     ✓ Created est_annual_revenue (PRIMARY) using calendar-based occupancy ({valid_rev.sum():,} valid values)")
        print(f"        Revenue = Price × (Booked + Host-blocked days)")
    
    # SECONDARY: Estimated annual revenue using actual booked days (for comparison)
    if 'price_clean' in df.columns and 'estimated_occupancy_l365d' in df.columns:
        df['est_annual_revenue_booked'] = df['price_clean'] * df['estimated_occupancy_l365d']
        valid_rev_booked = df['est_annual_revenue_booked'].notna()
        print(f"     ✓ Created est_annual_revenue_booked (SECONDARY) using actual booked days ({valid_rev_booked.sum():,} valid values)")
        print(f"        Revenue = Price × Booked days only")
    
    # Size-adjusted revenue metrics (using PRIMARY revenue)
    if 'est_annual_revenue' in df.columns:
        if 'accommodates' in df.columns:
            df['revenue_per_accommodates'] = df['est_annual_revenue'] / df['accommodates']
            print(f"     ✓ Created revenue_per_accommodates (unit economics)")
        
        if 'bedrooms' in df.columns:
            # Handle 0 bedrooms (studios)
            valid_br = (df['bedrooms'] > 0) & df['est_annual_revenue'].notna()
            df.loc[valid_br, 'revenue_per_bedroom'] = (
                df.loc[valid_br, 'est_annual_revenue'] / df.loc[valid_br, 'bedrooms']
            )
            print(f"     ✓ Created revenue_per_bedroom ({valid_br.sum():,} valid values)")
        
        # Log-transformed revenue for percentage analysis
        valid_rev = df['est_annual_revenue'] > 0
        df.loc[valid_rev, 'log_est_revenue'] = np.log(
            df.loc[valid_rev, 'est_annual_revenue']
        )
        print(f"     ✓ Created log_est_revenue ({valid_rev.sum():,} valid values)")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - PRIORITY 4: PROFESSIONALIZATION METRICS
# ============================================================================

def add_professionalization_metrics(df, city_name):
    """
    Priority 4: Add professionalization metrics based on host listing counts
    
    WHY:
    - Professional operators (multiple properties) behave differently than casual hosts
    - Professional markets may be more stable but also more competitive
    - Helps identify market maturity and investment opportunities
    - Professional hosts may have different pricing, occupancy, and quality strategies
    
    Listing-Level Metrics:
    - host_is_professional: Binary indicator (1 if host has 2+ listings in city)
    - host_listings_in_city: Number of listings host has in this city
    - host_professional_tier: Categorization (casual=1, small=2-5, medium=6-20, large=21+)
    
    Market-Level Metrics (calculated per city):
    - pct_professional_hosts: % of listings from hosts with 2+ properties
    - pct_large_operators: % of listings from hosts with 21+ properties
    - median_host_listings: Median number of listings per host
    - gini_host_concentration: Gini coefficient of host concentration (0=even, 1=monopoly)
    - market_professionalization_score: Composite score (0-100)
    
    Args:
        df: DataFrame with listing data
        city_name: Name of city (for display purposes)
    
    Returns:
        DataFrame with added professionalization metrics
    """
    print(f"\n  🔧 Priority 4: Adding professionalization metrics for {city_name}...")
    
    df = df.copy()
    
    # Use calculated_host_listings_count (city-specific) as primary metric
    if 'calculated_host_listings_count' in df.columns:
        # Convert to numeric if needed
        if df['calculated_host_listings_count'].dtype == 'object':
            df['calculated_host_listings_count'] = pd.to_numeric(
                df['calculated_host_listings_count'], errors='coerce'
            )
        
        # Listing-level professionalization
        df['host_listings_in_city'] = df['calculated_host_listings_count']
        
        # Binary professional indicator (2+ listings = professional)
        df['host_is_professional'] = (df['calculated_host_listings_count'] >= 2).astype(int)
        n_professional = df['host_is_professional'].sum()
        print(f"     ✓ Created host_is_professional ({n_professional:,} professional hosts, {n_professional/len(df)*100:.1f}%)")
        
        # Professional tier categorization
        def categorize_professional(count):
            if pd.isna(count) or count < 1:
                return 'unknown'
            elif count == 1:
                return 'casual'
            elif count <= 5:
                return 'small_professional'
            elif count <= 20:
                return 'medium_professional'
            else:
                return 'large_professional'
        
        df['host_professional_tier'] = df['calculated_host_listings_count'].apply(categorize_professional)
        tier_counts = df['host_professional_tier'].value_counts()
        print(f"     ✓ Created host_professional_tier")
        for tier, count in tier_counts.items():
            print(f"        {tier}: {count:,} listings ({count/len(df)*100:.1f}%)")
        
        # Market-level metrics (aggregated)
        total_listings = len(df)
        professional_listings = df['host_is_professional'].sum()
        large_operator_listings = (df['calculated_host_listings_count'] >= 21).sum()
        
        df['pct_professional_hosts'] = (professional_listings / total_listings) * 100
        df['pct_large_operators'] = (large_operator_listings / total_listings) * 100
        df['median_host_listings'] = df['calculated_host_listings_count'].median()
        
        # Gini coefficient for host concentration (measure of market concentration)
        # Gini = 0 means perfectly even distribution, 1 means one host owns everything
        # Calculate based on distribution of listings across hosts
        host_listing_counts = df['calculated_host_listings_count'].dropna()
        if len(host_listing_counts) > 1:
            # Sort in ascending order
            sorted_counts = np.sort(host_listing_counts.values)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            total = cumsum[-1]
            
            if total > 0:
                # Gini coefficient formula
                gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * total) - (n + 1) / n
                gini = abs(gini)  # Ensure positive, bound between 0 and 1
                gini = min(gini, 1.0)
            else:
                gini = 0
        else:
            gini = 0
        
        df['gini_host_concentration'] = gini
        
        # Composite professionalization score (0-100)
        # Based on: % professional, % large operators, median listings, Gini coefficient
        pct_prof = df['pct_professional_hosts'].iloc[0]
        pct_large = df['pct_large_operators'].iloc[0]
        median_listings = df['median_host_listings'].iloc[0]
        
        # Normalize components (0-100 scale)
        # % professional: already 0-100
        # % large operators: scale to 0-100 (assuming max is ~50%)
        pct_large_scaled = min(pct_large * 2, 100)
        # Median listings: scale assuming max is ~50
        median_scaled = min(median_listings / 50 * 100, 100)
        # Gini: already 0-1, scale to 0-100
        gini_scaled = gini * 100
        
        # Weighted average (can adjust weights)
        professionalization_score = (
            0.3 * pct_prof +
            0.3 * pct_large_scaled +
            0.2 * median_scaled +
            0.2 * gini_scaled
        )
        
        df['market_professionalization_score'] = professionalization_score
        
        print(f"     ✓ Market-level metrics:")
        print(f"        % Professional hosts: {pct_prof:.1f}%")
        print(f"        % Large operators (21+): {pct_large:.1f}%")
        print(f"        Median listings per host: {median_listings:.1f}")
        print(f"        Gini coefficient: {gini:.3f}")
        print(f"        Professionalization score: {professionalization_score:.1f}/100")
    
    return df


# ============================================================================
# FEATURE ENGINEERING - ZILLOW PRICE INTEGRATION
# ============================================================================

def add_zillow_prices(df, city_name, zillow_data=None):
    """
    Add Zillow purchase prices to listings based on metro matching.
    
    Args:
        df: DataFrame with listing data
        city_name: Name of Airbnb city (e.g., "New_York", "Austin")
        zillow_data: DataFrame from load_all_zillow_data() (if None, will load)
    
    Returns:
        DataFrame with added columns: purchase_price, monthly_payment, zori_rent
    """
    print(f"\n  🔧 Adding Zillow purchase prices for {city_name}...")
    
    # Load Zillow data if not provided
    if zillow_data is None:
        try:
            from load_zillow_data import load_all_zillow_data
            zillow_data = load_all_zillow_data()
        except Exception as e:
            print(f"     ⚠️  Could not load Zillow data: {e}")
            return df
    
    # Match city to Zillow metro
    try:
        from match_cities_to_zillow import match_city_to_zillow
        metro_data = match_city_to_zillow(city_name, zillow_data)
        
        if metro_data is None:
            print(f"     ⚠️  No Zillow data available for {city_name}")
            return df
        
        # Add metro-level prices to all listings
        df['purchase_price'] = metro_data['zhvi_price']
        df['monthly_payment'] = metro_data['monthly_payment']
        df['zori_rent'] = metro_data['zori_rent']
        df['zillow_metro'] = metro_data['RegionName']
        
        n_valid = df['purchase_price'].notna().sum()
        print(f"     ✓ Added Zillow prices from {metro_data['RegionName']}")
        print(f"        Purchase price: ${metro_data['zhvi_price']:,.0f}")
        if pd.notna(metro_data['monthly_payment']):
            print(f"        Monthly payment: ${metro_data['monthly_payment']:,.0f}")
        print(f"        Applied to {n_valid:,} listings")
        
    except Exception as e:
        print(f"     ⚠️  Error matching city to Zillow: {e}")
        return df
    
    return df


def add_roi_metrics(df):
    """
    Calculate ROI metrics using Zillow purchase prices and Airbnb revenue.
    
    Calculates ROI for BOTH primary (calendar-based) and secondary (booked days) revenue metrics
    to enable comparison and show profitability under both scenarios.
    
    PRIMARY Metrics (using calendar-based occupancy):
    - annual_cash_flow: est_annual_revenue - (monthly_payment * 12)
    - cash_on_cash_roi: annual_cash_flow / (purchase_price * 0.20) [20% downpayment]
    - cap_rate: est_annual_revenue / purchase_price [gross cap rate]
    - price_to_rent_ratio: purchase_price / (est_annual_revenue / 12) [months to pay off]
    - revenue_yield: est_annual_revenue / purchase_price [annual revenue as % of purchase]
    
    SECONDARY Metrics (using booked days only, for comparison):
    - annual_cash_flow_booked: est_annual_revenue_booked - (monthly_payment * 12)
    - cash_on_cash_roi_booked: annual_cash_flow_booked / (purchase_price * 0.20)
    - cap_rate_booked: est_annual_revenue_booked / purchase_price
    - revenue_yield_booked: est_annual_revenue_booked / purchase_price
    
    Args:
        df: DataFrame with purchase_price, monthly_payment, est_annual_revenue, est_annual_revenue_booked
    
    Returns:
        DataFrame with added ROI metric columns (both primary and secondary)
    """
    print(f"\n  🔧 Adding ROI metrics...")
    print(f"     Calculating ROI for both PRIMARY (calendar-based) and SECONDARY (booked days) revenue")
    
    df = df.copy()
    
    # Check if we have the required data
    has_price = 'purchase_price' in df.columns and df['purchase_price'].notna().any()
    has_revenue = 'est_annual_revenue' in df.columns and df['est_annual_revenue'].notna().any()
    has_revenue_booked = 'est_annual_revenue_booked' in df.columns and df['est_annual_revenue_booked'].notna().any()
    has_payment = 'monthly_payment' in df.columns and df['monthly_payment'].notna().any()
    
    if not has_price or not has_revenue:
        print(f"     ⚠️  Missing required data for ROI calculations")
        if not has_price:
            print(f"        Missing: purchase_price")
        if not has_revenue:
            print(f"        Missing: est_annual_revenue")
        return df
    
    # ============================================================================
    # PRIMARY ROI METRICS (using calendar-based occupancy)
    # ============================================================================
    
    # Annual cash flow (revenue - annual payments)
    if has_payment:
        df['annual_cash_flow'] = df['est_annual_revenue'] - (df['monthly_payment'] * 12)
        valid_cf = df['annual_cash_flow'].notna()
        print(f"     ✓ Created annual_cash_flow (PRIMARY) ({valid_cf.sum():,} valid values)")
    else:
        # If no monthly payment data, cash flow = revenue (simplified)
        df['annual_cash_flow'] = df['est_annual_revenue']
        print(f"     ✓ Created annual_cash_flow (PRIMARY, simplified, no payment data)")
    
    # Cash-on-cash ROI (assuming 20% downpayment)
    # ROI = Annual Cash Flow / Down Payment
    downpayment = df['purchase_price'] * 0.20
    valid_roi = (df['annual_cash_flow'].notna() & (downpayment > 0))
    df.loc[valid_roi, 'cash_on_cash_roi'] = (
        df.loc[valid_roi, 'annual_cash_flow'] / downpayment.loc[valid_roi]
    )
    # Convert to percentage
    df.loc[valid_roi, 'cash_on_cash_roi'] = df.loc[valid_roi, 'cash_on_cash_roi'] * 100
    print(f"     ✓ Created cash_on_cash_roi (PRIMARY) ({valid_roi.sum():,} valid values)")
    
    # Cap rate (gross cap rate = revenue / purchase price)
    valid_cap = (df['est_annual_revenue'].notna() & (df['purchase_price'] > 0))
    df.loc[valid_cap, 'cap_rate'] = (
        df.loc[valid_cap, 'est_annual_revenue'] / df.loc[valid_cap, 'purchase_price']
    )
    # Convert to percentage
    df.loc[valid_cap, 'cap_rate'] = df.loc[valid_cap, 'cap_rate'] * 100
    print(f"     ✓ Created cap_rate (PRIMARY) ({valid_cap.sum():,} valid values)")
    
    # Price-to-rent ratio (months to pay off at current revenue)
    monthly_revenue = df['est_annual_revenue'] / 12
    valid_ptr = (monthly_revenue.notna() & (monthly_revenue > 0) & (df['purchase_price'] > 0))
    df.loc[valid_ptr, 'price_to_rent_ratio'] = (
        df.loc[valid_ptr, 'purchase_price'] / monthly_revenue.loc[valid_ptr]
    )
    print(f"     ✓ Created price_to_rent_ratio (PRIMARY) ({valid_ptr.sum():,} valid values)")
    
    # Revenue yield (annual revenue as % of purchase price, same as cap rate but different interpretation)
    df.loc[valid_cap, 'revenue_yield'] = df.loc[valid_cap, 'cap_rate']
    print(f"     ✓ Created revenue_yield (PRIMARY) ({valid_cap.sum():,} valid values)")
    
    # ============================================================================
    # SECONDARY ROI METRICS (using booked days only, for comparison)
    # ============================================================================
    
    if has_revenue_booked:
        # Annual cash flow (booked days)
        if has_payment:
            df['annual_cash_flow_booked'] = df['est_annual_revenue_booked'] - (df['monthly_payment'] * 12)
        else:
            df['annual_cash_flow_booked'] = df['est_annual_revenue_booked']
        valid_cf_booked = df['annual_cash_flow_booked'].notna()
        print(f"     ✓ Created annual_cash_flow_booked (SECONDARY) ({valid_cf_booked.sum():,} valid values)")
        
        # Cash-on-cash ROI (booked days)
        valid_roi_booked = (df['annual_cash_flow_booked'].notna() & (downpayment > 0))
        df.loc[valid_roi_booked, 'cash_on_cash_roi_booked'] = (
            df.loc[valid_roi_booked, 'annual_cash_flow_booked'] / downpayment.loc[valid_roi_booked]
        ) * 100
        print(f"     ✓ Created cash_on_cash_roi_booked (SECONDARY) ({valid_roi_booked.sum():,} valid values)")
        
        # Cap rate (booked days)
        valid_cap_booked = (df['est_annual_revenue_booked'].notna() & (df['purchase_price'] > 0))
        df.loc[valid_cap_booked, 'cap_rate_booked'] = (
            df.loc[valid_cap_booked, 'est_annual_revenue_booked'] / df.loc[valid_cap_booked, 'purchase_price']
        ) * 100
        print(f"     ✓ Created cap_rate_booked (SECONDARY) ({valid_cap_booked.sum():,} valid values)")
        
        # Revenue yield (booked days)
        df.loc[valid_cap_booked, 'revenue_yield_booked'] = df.loc[valid_cap_booked, 'cap_rate_booked']
        print(f"     ✓ Created revenue_yield_booked (SECONDARY) ({valid_cap_booked.sum():,} valid values)")
    
    # Summary statistics (PRIMARY)
    if valid_roi.any():
        median_roi = df.loc[valid_roi, 'cash_on_cash_roi'].median()
        print(f"        Median cash-on-cash ROI (PRIMARY): {median_roi:.1f}%")
    
    if valid_cap.any():
        median_cap = df.loc[valid_cap, 'cap_rate'].median()
        print(f"        Median cap rate (PRIMARY): {median_cap:.2f}%")
    
    # Summary statistics (SECONDARY, if available)
    if has_revenue_booked:
        if 'cash_on_cash_roi_booked' in df.columns:
            valid_roi_booked = df['cash_on_cash_roi_booked'].notna()
            if valid_roi_booked.any():
                median_roi_booked = df.loc[valid_roi_booked, 'cash_on_cash_roi_booked'].median()
                print(f"        Median cash-on-cash ROI (SECONDARY): {median_roi_booked:.1f}%")
        
        if 'cap_rate_booked' in df.columns:
            valid_cap_booked = df['cap_rate_booked'].notna()
            if valid_cap_booked.any():
                median_cap_booked = df.loc[valid_cap_booked, 'cap_rate_booked'].median()
                print(f"        Median cap rate (SECONDARY): {median_cap_booked:.2f}%")
    
    return df


# ============================================================================
# MASTER FEATURE ENGINEERING FUNCTION
# ============================================================================

def apply_all_feature_engineering(df, city_name, include_zillow=True):
    """
    Apply all feature engineering transformations in sequence
    
    This is the main function that orchestrates all transformations.
    Call this after loading raw data and before analysis.
    
    Args:
        df: Raw DataFrame from CSV
        city_name: Name of city (for display and within-city metrics)
    
    Returns:
        DataFrame with all engineered features added
    """
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING FOR {city_name.upper()}")
    print(f"{'='*80}")
    print(f"Starting with {len(df):,} listings and {len(df.columns)} columns")
    
    # Exclude hotel rooms from analysis
    # Rationale: Hotel rooms represent commercial operations, not residential
    # rental investments, which is the focus of this analysis
    if 'room_type' in df.columns:
        n_hotel = (df['room_type'] == 'Hotel room').sum()
        if n_hotel > 0:
            df = df[df['room_type'] != 'Hotel room'].copy()
            print(f"  🏨 Excluded {n_hotel:,} hotel room listings from analysis")
            print(f"     Focus: Residential rental investments (Entire home/apt, Private room, Shared room)")
    
    print(f"After filtering: {len(df):,} listings")
    
    original_cols = len(df.columns)
    
    # Apply transformations in priority order
    df = add_log_transforms(df)
    df = add_within_city_metrics(df, city_name)
    df = add_revenue_proxies(df)
    df = add_professionalization_metrics(df, city_name)
    
    # Add Zillow prices and ROI metrics (if requested)
    if include_zillow:
        df = add_zillow_prices(df, city_name)
        df = add_roi_metrics(df)
    
    new_cols = len(df.columns)
    added_cols = new_cols - original_cols
    
    print(f"\n  ✅ Feature engineering complete!")
    print(f"     Added {added_cols} new columns (now {new_cols} total)")
    print(f"{'='*80}")
    
    return df


# ============================================================================
# EXISTING HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

def sanitize_text(text, max_length=MAX_TEXT_LENGTH):
    """
    Sanitize text for safe CSV storage
    - Truncate to max_length
    - Remove newlines and extra spaces
    - Remove problematic characters
    """
    if pd.isna(text) or text is None:
        return text
    
    text = str(text)
    
    # Remove newlines and tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove <br /> tags
    text = text.replace('<br />', ' ').replace('<br>', ' ')
    
    # Collapse multiple spaces
    text = ' '.join(text.split())
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + '...'
    
    return text

def analyze_variable(series, var_name):
    """
    Comprehensive analysis of a single variable
    
    Returns dict with: name, type, range, min, max, mean, median, std, precision,
    and for discrete: mode, n_unique
    """
    stats = {
        'variable_name': var_name,
        'data_type': None,
        'range': None,
        'min': None,
        'max': None,
        'mean': None,
        'median': None,
        'std': None,
        'precision': None,
        'mode': None,
        'n_unique': None,
        'n_missing': series.isna().sum(),
        'pct_missing': (series.isna().sum() / len(series) * 100).round(2)
    }
    
    # Remove missing values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        stats['data_type'] = 'empty'
        return stats
    
    # Determine if continuous or discrete
    if pd.api.types.is_numeric_dtype(clean_series):
        n_unique = clean_series.nunique()
        
        # Heuristic: if fewer than 20 unique values or all integers, treat as discrete
        if n_unique < 20 or (clean_series.dtype in ['int64', 'int32'] and n_unique < 50):
            stats['data_type'] = 'discrete_numeric'
        else:
            stats['data_type'] = 'continuous'
        
        # Calculate statistics
        stats['min'] = clean_series.min()
        stats['max'] = clean_series.max()
        stats['range'] = stats['max'] - stats['min']
        stats['mean'] = clean_series.mean()
        stats['median'] = clean_series.median()
        stats['std'] = clean_series.std()
        
        # Precision (number of decimal places)
        if stats['data_type'] == 'continuous':
            # Sample some values to determine precision
            sample_vals = clean_series.head(100).astype(str)
            decimal_places = [len(str(v).split('.')[-1]) if '.' in str(v) else 0 for v in sample_vals]
            stats['precision'] = max(decimal_places) if decimal_places else 0
        else:
            stats['precision'] = 0
        
        # Mode and unique count
        stats['mode'] = clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else None
        stats['n_unique'] = n_unique
        
    else:
        # Categorical/text variable
        stats['data_type'] = 'discrete_categorical'
        stats['n_unique'] = clean_series.nunique()
        
        # Sanitize text values for min, max, mode
        mode_val = clean_series.mode().iloc[0] if len(clean_series.mode()) > 0 else None
        stats['mode'] = sanitize_text(mode_val, max_length=50)
        
        # For categorical, min/max are first/last alphabetically
        stats['min'] = sanitize_text(clean_series.min(), max_length=50)
        stats['max'] = sanitize_text(clean_series.max(), max_length=50)
    
    return stats

def create_variable_summary_table(df, city_name):
    """Create comprehensive variable summary table for a city"""
    print(f"\n{'='*80}")
    print(f"ANALYZING VARIABLES FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Analyze each variable
    all_stats = []
    for col in df.columns:
        if col != 'city':  # Skip city identifier
            print(f"  Analyzing: {col}")
            stats = analyze_variable(df[col], col)
            all_stats.append(stats)
    
    # Create DataFrame
    summary_df = pd.DataFrame(all_stats)
    
    # Reorder columns for better readability
    col_order = ['variable_name', 'data_type', 'n_unique', 'n_missing', 'pct_missing',
                 'min', 'max', 'range', 'mean', 'median', 'std', 'precision', 'mode']
    summary_df = summary_df[col_order]
    
    return summary_df

def create_all_correlation_matrices(df, city_name, output_dir, top_n=25):
    """
    Create correlation matrices for all combinations of numeric variables
    
    Args:
        df: DataFrame with city data
        city_name: Name of the city
        output_dir: Directory to save outputs
        top_n: Number of top correlations to analyze (default 25)
    """
    print(f"\n{'='*80}")
    print(f"CREATING CORRELATION MATRICES FOR {city_name.upper()}")
    print(f"{'='*80}")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'city']
    
    if len(numeric_cols) < 2:
        print(f"  ⚠️  Not enough numeric variables for correlation analysis")
        return
    
    print(f"  Found {len(numeric_cols)} numeric variables")
    print(f"  Variables: {', '.join(numeric_cols)}")
    
    # Create correlation matrix for all numeric variables
    print(f"\n  Creating full correlation matrix...")
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix as CSV
    corr_matrix.to_csv(output_dir / f'{city_name}_correlation_matrix.csv')
    print(f"  ✓ Saved correlation matrix CSV")
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'{city_name.upper()} - Correlation Matrix (All Variables)', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{city_name}_correlation_heatmap_full.png', 
               dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved full correlation heatmap")
    plt.close()
    
    # Get top correlations
    print(f"\n  Extracting top {top_n} correlations...")
    
    # Get upper triangle of correlation matrix (to avoid duplicates)
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Stack and sort
    corr_pairs = corr_upper.stack().sort_values(ascending=False)
    
    # Get top N positive and top N negative correlations
    top_positive = corr_pairs.head(top_n)
    top_negative = corr_pairs.tail(top_n)
    
    # Combine and create DataFrame
    top_corrs = pd.concat([top_positive, top_negative])
    top_corrs_df = pd.DataFrame({
        'variable_1': [pair[0] for pair in top_corrs.index],
        'variable_2': [pair[1] for pair in top_corrs.index],
        'correlation': top_corrs.values
    })
    
    # Save top correlations
    top_corrs_df.to_csv(output_dir / f'{city_name}_top_correlations.csv', index=False)
    print(f"  ✓ Saved top {len(top_corrs_df)} correlations")
    
    # Create scatter plots for top correlations
    print(f"\n  Creating scatter plots for top correlations...")
    n_plots = min(9, len(top_corrs_df))  # Up to 9 plots (3x3 grid)
    
    if n_plots > 0:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(n_plots):
            row = top_corrs_df.iloc[idx]
            var1, var2, corr = row['variable_1'], row['variable_2'], row['correlation']
            
            # Create scatter plot
            axes[idx].scatter(df[var1], df[var2], alpha=0.3, s=20)
            axes[idx].set_xlabel(var1, fontsize=9)
            axes[idx].set_ylabel(var2, fontsize=9)
            axes[idx].set_title(f'r = {corr:.3f}', fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, 9):
            axes[idx].axis('off')
        
        plt.suptitle(f'{city_name.upper()} - Top Correlation Scatter Plots', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / f'{city_name}_correlation_scatter_plots.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved scatter plots")
        plt.close()


# ============================================================================
# MAIN CITY ANALYSIS FUNCTION (MODIFIED TO INCLUDE FEATURE ENGINEERING)
# ============================================================================

def analyze_city(city_name, base_dir='.', use_detailed=False):
    """
    Analyze a single city with feature engineering
    
    Modified to include Priority 1-3 feature engineering before analysis
    """
    city_path = Path(base_dir) / city_name
    
    if not city_path.exists():
        print(f"⚠️  Directory not found: {city_path}")
        return None
    
    # Determine which file to use
    if use_detailed:
        listings_file = city_path / 'listings.csv.gz'
        if not listings_file.exists():
            print(f"⚠️  Detailed file (listings.csv.gz) not found for {city_name}")
            print(f"    Falling back to simple listings.csv...")
            listings_file = city_path / 'listings.csv'
            if listings_file.exists():
                print(f"📊 Using SIMPLE dataset (19 variables) as fallback")
        else:
            print(f"📊 Using DETAILED dataset (79 variables) from listings.csv.gz")
    else:
        listings_file = city_path / 'listings.csv'
        if not listings_file.exists():
            print(f"⚠️  Simple file (listings.csv) not found for {city_name}")
            print(f"    Trying detailed listings.csv.gz...")
            listings_file = city_path / 'listings.csv.gz'
            if listings_file.exists():
                print(f"📊 Using DETAILED dataset (79 variables) as fallback")
        else:
            print(f"📊 Using SIMPLE dataset (19 variables) from listings.csv")
    
    if not listings_file.exists():
        print(f"❌ No listings file found for {city_name}")
        return None
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    try:
        if str(listings_file).endswith('.gz'):
            df = pd.read_csv(listings_file, compression='gzip')
        else:
            df = pd.read_csv(listings_file)
        print(f"✓ Loaded {len(df):,} listings with {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading {city_name}: {e}")
        return None
    
    # Clean price if present
    if 'price' in df.columns:
        df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
    
    # *** NEW: APPLY FEATURE ENGINEERING ***
    df = apply_all_feature_engineering(df, city_name)
    
    # Create output directory in city folder
    output_dir = city_path / 'analysis_output'
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create variable summary table
    print(f"\n📋 Creating variable summary table...")
    summary_table = create_variable_summary_table(df, city_name)
    summary_table.to_csv(output_dir / f'{city_name}_variable_summary.csv', index=False)
    print(f"✓ Saved: {city_name}_variable_summary.csv")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print(f"VARIABLE SUMMARY TABLE - {city_name.upper()}")
    print(f"{'='*80}")
    print(summary_table.to_string(index=False))
    
    # 2. Create correlation matrices
    create_all_correlation_matrices(df, city_name, output_dir, top_n=TOP_CORRELATIONS_N)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE FOR {city_name.upper()}")
    print(f"{'='*80}")
    print(f"Output saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - {city_name}_variable_summary.csv")
    print(f"  - {city_name}_correlation_matrix.csv")
    print(f"  - {city_name}_correlation_heatmap_full.png")
    print(f"  - {city_name}_top_correlations.csv")
    print(f"  - {city_name}_correlation_scatter_plots.png")
    
    return summary_table

def analyze_all_cities(city_folders, base_dir='.', use_detailed=False):
    """
    Run analysis for all specified cities
    
    Args:
        city_folders: List of city folder names
        base_dir: Base directory
        use_detailed: If True, use detailed 79-variable datasets
    """
    print(f"\n{'#'*80}")
    print(f"CITY-LEVEL ANALYSIS SCRIPT - WITH FEATURE ENGINEERING")
    print(f"{'#'*80}")
    
    if use_detailed:
        print(f"\n🔍 MODE: DETAILED ANALYSIS (79 variables from listings_csv.gz)")
    else:
        print(f"\n🔍 MODE: SIMPLE ANALYSIS (19 variables from listings.csv)")
    
    print(f"\nWill analyze {len(city_folders)} cities")
    print(f"Cities: {', '.join(city_folders)}")
    
    results = {}
    successful = 0
    failed = 0
    
    for city in city_folders:
        try:
            summary = analyze_city(city, base_dir, use_detailed=use_detailed)
            if summary is not None:
                results[city] = summary
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ ERROR analyzing {city}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"{'#'*80}")
    print(f"✅ Successfully analyzed: {successful} cities")
    print(f"❌ Failed: {failed} cities")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this script from your main Airbnb_Data directory
    
    Directory structure should be:
    Airbnb_Data/
        city_level_analysis.py  (this file)
        Austin/
            listings.csv        (19 variables - simple)
            listings.csv.gz     (79 variables - detailed)  ⚠️ NOTE: .csv.gz not _csv.gz
        Boston/
            listings.csv
            listings.csv.gz
        ... etc
    
    Usage:
        python city_level_analysis.py                    # All cities, simple datasets
        python city_level_analysis.py -all               # All cities, detailed datasets
        python city_level_analysis.py Austin             # Just Austin, simple dataset
        python city_level_analysis.py Austin -all        # Just Austin, detailed dataset
        python city_level_analysis.py -all Austin        # Order doesn't matter
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    
    # Check if a specific city was requested
    city_args = [arg for arg in sys.argv[1:] if arg != '-all']
    single_city = city_args[0] if city_args else None
    
    # ====== CUSTOMIZE THIS LIST ======
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    # =================================
    
    # Determine which cities to analyze
    if single_city:
        # Single city mode (case-insensitive matching)
        city_folders = [c for c in all_cities if c.lower() == single_city.lower()]
        
        if not city_folders:
            print(f"\n❌ ERROR: City '{single_city}' not found in city list!")
            print(f"\nAvailable cities:")
            for city in all_cities:
                print(f"  - {city}")
            print(f"\nUsage: python city_level_analysis.py CityName [-all]")
            sys.exit(1)
        
        print(f"\n🎯 SINGLE CITY MODE: Analyzing {city_folders[0]} only")
        print(f"   (For all cities, run without city name)")
    else:
        # All cities mode
        city_folders = all_cities
        print(f"\n📊 BATCH MODE: Analyzing all {len(city_folders)} cities")
    
    # Run analysis
    results = analyze_all_cities(city_folders, base_dir='.', use_detailed=use_detailed)
    
    print(f"\n{'#'*80}")
    print(f"ALL DONE! Check each city's 'analysis_output' folder for results.")
    print(f"{'#'*80}")
    
    # Verify what was actually analyzed
    if results:
        sample_city = list(results.keys())[0]
        num_vars = len(results[sample_city])
        print(f"\n✅ VERIFICATION: Analyzed {num_vars} variables per city")
        
        if num_vars >= 70:
            print(f"   🎯 SUCCESS: Full detailed analysis with ~79+ variables")
        elif num_vars <= 25:
            print(f"   ⚠️  Simple analysis with ~19+ variables")
            print(f"   💡 TIP: Run 'python city_level_analysis.py -all' for full 79-variable analysis")
        else:
            print(f"   ⚠️  Partial dataset detected")
    
    if use_detailed:
        print(f"\n💡 You requested DETAILED analysis (-all flag)")
    else:
        print(f"\n💡 You used SIMPLE analysis (default)")
        print(f"   To get full analysis with 79 variables, run: python city_level_analysis.py -all")