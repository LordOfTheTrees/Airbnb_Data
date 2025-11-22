nvestment Analysis Framework - Using Zillow Purchase Price Data
Goal
Identify best rental property types and locations for purchase using Airbnb revenue data + Zillow purchase price data. Focus on short-term cash flow (1-2 years).

Strategy
Use actual purchase prices from Zillow (ZHVI - Zillow Home Value Index) combined with Airbnb revenue data to calculate real ROI metrics. Match Airbnb cities to Zillow metro areas.

---

Phase 1: Zillow Data Integration
1.1 Load and Process Zillow Data
New File: load_zillow_data.py

Functions:

load_zillow_zhvi(): Load home value index (purchase prices) by metro
load_zillow_zori(): Load rental index (for comparison/validation)
load_zillow_monthly_payment(): Load monthly payment data (20% down, middle tier)
Extract most recent month (2025-10 or latest available) for each metro
Return DataFrame with: RegionName, StateName, zhvi_price, zori_rent, monthly_payment
1.2 City-to-Metro Matching
New File: match_cities_to_zillow.py

Create mapping dictionary for Airbnb cities → Zillow metros:

Direct matches: "Austin" → "Austin, TX", "Chicago" → "Chicago, IL", etc.
Name transformations: "New_York" → "New York, NY", "Los_Angeles" → "Los Angeles, CA"
Sub-city to metro: "Jersey_City" → "New York, NY", "Cambridge" → "Boston, MA", "Oakland" → "San Francisco, CA"
Handle special cases: "Hawaii" → check for Honolulu metro?, "Oregon" → "Portland, OR"?, "Rhode_Island" → "Providence, RI"?
Handle unmatchable cities: "Paris" (France - no Zillow data)
Output: Dictionary mapping each Airbnb city to Zillow metro (or None if unmatchable)

1.3 Merge Zillow Prices with Airbnb Data
File: city_level_analysis.py - Add new function add_zillow_prices()

For each listing:

Match city to Zillow metro
Assign metro-level purchase price (ZHVI) to listing
Calculate: purchase_price = zhvi_price (metro-level, not property-specific)
Note: This is a limitation - we have metro-level prices, not property-specific
---

Phase 2: ROI Calculations with Real Purchase Prices
2.1 ROI Metrics
File: city_level_analysis.py - Add to add_revenue_proxies() or new function

Calculate real ROI metrics:

annual_cash_flow: est_annual_revenue - (monthly_payment * 12) (simplified, assumes monthly_payment covers P&I)
cash_on_cash_roi: annual_cash_flow / (purchase_price * 0.20) (20% downpayment)
cap_rate: est_annual_revenue / purchase_price (gross cap rate)
price_to_rent_ratio: purchase_price / (est_annual_revenue / 12) (months to pay off)
revenue_yield: est_annual_revenue / purchase_price (annual revenue as % of purchase price)
2.2 Property Type × Size Segment Analysis
New File: analyze_property_segments.py

Segment by:

Property type (Entire home/apt, Private room, Shared room)
Size bins (Studio, 1BR, 2BR, 3+BR) using bedrooms
Calculate for each segment:
Median cash_on_cash_roi
Median cap_rate
Median revenue_yield
Median occupancy rate
Count of listings (market size)
Output: Ranking table showing best segments by ROI
---

Phase 3: Neighborhood-Level Profitability Analysis
3.1 Geographic Aggregation with ROI
New File: analyze_neighborhood_profitability.py

Use available geographic fields:

Primary: zipcode (if available and complete)
Fallback: neighbourhood_cleansed (geocoded neighborhoods)
Fallback 2: neighbourhood_group_cleansed (larger areas)
For each geographic unit within each city:

Calculate median ROI metrics (cash_on_cash_roi, cap_rate)
Calculate median revenue_yield
Calculate listing density
Identify "hotspots": neighborhoods with high ROI + high occupancy
3.2 Geographic Visualization
Create heatmaps showing:

Cash-on-cash ROI by zipcode/neighborhood
Cap rates by geography
Revenue yield by geography
Output: neighborhood_roi_maps.png per city
---

Phase 4: Emerging Market Indicators
4.1 Review Velocity Metrics
File: city_level_analysis.py - Add new function add_growth_indicators()

Create growth proxies from review data:

reviews_per_month: number_of_reviews / months_since_first_review
recent_review_velocity: Reviews in last 90 days (if last_review is recent)
review_growth_indicator: High reviews_per_month + recent activity = growing demand
listing_age_months: Months since host_since (newer listings = emerging)
4.2 Market Maturity Indicators
New File: analyze_market_maturity.py


Implementation Order
Phase 1: Zillow data loading + city matching (foundation)
Phase 2: ROI calculations + property segment analysis (within-city)
Phase 3: Neighborhood-level ROI analysis (within-city)
Phase 4: review velocity metrics
Phase 5+: DEFERRED - Needs refinement before proceeding, will be around 
---

Key Files to Create/Modify
New Files:

load_zillow_data.py - Load Zillow ZHVI, ZORI, monthly payment data
match_cities_to_zillow.py - Map Airbnb cities to Zillow metros
analyze_property_segments.py - Property type × size ROI analysis (within-city)
analyze_neighborhood_profitability.py - Geographic ROI analysis (within-city)
analyze_neighborhood_growth.py - Within-city neighborhood growth analysis
surface_city_comparison.py - Optional surface-level cross-city comparison
Modify Existing:

city_level_analysis.py - Add add_zillow_prices() and add_growth_indicators() functions
city_level_analysis.py - Enhance add_revenue_proxies() with ROI metrics
---

Output Deliverables
Property Segment Rankings: Best property types by ROI (cash-on-cash, cap rate) - per city
Neighborhood ROI Heatmaps: Geographic ROI visualization - per city
Review Velocity Metrics: Growth indicators for listings and neighborhoods - per city
---

Limitations & Notes
Metro-level prices: Zillow data is at metro level, not property-specific. All listings in a metro get the same purchase price.
Property type mismatch: Zillow ZHVI is for single-family homes and condos, but Airbnb includes many property types (apartments, rooms, etc.)
Unmatchable cities: Some Airbnb cities (e.g., "Paris") have no Zillow data
Sub-city matching: Some Airbnb cities are sub-cities within metros (e.g., "Jersey_City" uses "New York, NY" metro prices)