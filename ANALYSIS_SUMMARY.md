# Airbnb Investment Analysis - Comprehensive Summary

**Project Goal**: Identify the best types of rental properties to purchase using Airbnb revenue data combined with Zillow purchase price data, focusing on short-term cash flow (1-2 years).

**Date**: 2025
**Data Coverage**: 22 cities across the United States (and Paris, France)

---

## Table of Contents

1. [Data Sources and Structure](#data-sources-and-structure)
2. [Data Exploration and Evaluation](#data-exploration-and-evaluation)
3. [Data Transformations](#data-transformations)
4. [Computed Metrics](#computed-metrics)
5. [Occupancy Metric Rationale](#occupancy-metric-rationale)
6. [Profitability Analysis](#profitability-analysis)
7. [City-Level Analysis](#city-level-analysis)
8. [Key Findings](#key-findings)
9. [Visualization Suite](#visualization-suite)
10. [Methodology Notes](#methodology-notes)

---

## 1. Data Sources and Structure

### Primary Data Sources

**Airbnb Listings Data** (Inside Airbnb)
- **Source**: Inside Airbnb scrapes (June-October 2025)
- **Coverage**: 22 cities, ~200,000+ listings total
- **Key Files per City**:
  - `listings.csv.gz`: Core listing data (79 variables in detailed dataset, 19 in simple)
  - `calendar.csv.gz`: Availability calendar data
  - `reviews.csv.gz`: Review data

**Zillow Real Estate Data**
- **Source**: Zillow Research Data (October 2025)
- **Coverage**: 894 metro areas
- **Key Metrics**:
  - **ZHVI** (Zillow Home Value Index): Metro-level purchase prices
  - **ZORI** (Zillow Rent Index): Metro-level rental rates
  - **Monthly Payment**: Calculated for 20% downpayment, middle tier (33rd-67th percentile)

### Data Structure

**Cities Analyzed**:
- Albany, Asheville, Austin, Bozeman, Cambridge
- Chicago, Columbus, Dallas, Denver, Hawaii
- Jersey_City, Los_Angeles, Nashville, New_Orleans
- New_York, Oakland, Oregon, Paris
- Rhode_Island, San_Francisco, Seattle, Washington_DC

**Data Collection Timeline**:
- All cities scraped between June and October 2025
- Time span: 111 days
- **No inflation adjustment needed** - all data from same time period

---

## 2. Data Exploration and Evaluation

### Initial Data Quality Assessment

**Price Data**:
- Raw `price` field contains currency symbols and commas
- Created `price_clean`: Numeric price after removing formatting
- Missing price data: ~30% of listings (varies by city)
- Price range: $10 - $10,000+ per night

**Occupancy Data**:
- `availability_365`: Calendar availability (days available in next 365 days)
- `estimated_occupancy_l365d`: **Raw field from Airbnb data** - Actual booked days in last 365 days (from Airbnb booking data). **NOT created by us** - comes directly from `listings.csv.gz`. Capped at 255 days (69.9%) due to 8-bit integer limitation
- This cap creates artificial ceiling and downward bias in analysis

**Size Attributes**:
- `accommodates`: Maximum guest capacity
- `bedrooms`: Number of bedrooms (0 = studio)
- `beds`: Number of beds
- `bathrooms`: Number of bathrooms

**Property Size vs. Price Relationships (Linear Regression Analysis)**:

To understand how property size attributes affect pricing, we performed linear regression analysis on log-transformed prices against each size metric. Outliers were filtered (bathrooms > 8, bedrooms > 10, beds > 20) to focus on typical residential properties.

**Example Results (Austin, TX)**:
- **Accommodates vs. Log Price**: 
  - Equation: `y = 0.1480x + 4.2362`
  - R² = 0.4082 (r = 0.6389)
  - **Interpretation**: Each additional guest capacity increases log(price) by 0.1480, meaning approximately 15.9% price increase per additional guest (e^(0.1480) - 1 ≈ 0.159)

- **Bathrooms vs. Log Price**:
  - Equation: `y = 0.5453x + 4.0778`
  - R² = 0.4075 (r = 0.6383)
  - **Interpretation**: Each additional bathroom increases log(price) by 0.5453, meaning approximately 72.5% price increase per additional bathroom (e^(0.5453) - 1 ≈ 0.725)

- **Bedrooms vs. Log Price**:
  - Equation: `y = 0.3591x + 4.2658`
  - R² = 0.3790 (r = 0.6156)
  - **Interpretation**: Each additional bedroom increases log(price) by 0.3591, meaning approximately 43.2% price increase per additional bedroom (e^(0.3591) - 1 ≈ 0.432)

- **Beds vs. Log Price**:
  - Equation: `y = 0.1911x + 4.4608`
  - R² = 0.3196 (r = 0.5653)
  - **Interpretation**: Each additional bed increases log(price) by 0.1911, meaning approximately 21.1% price increase per additional bed (e^(0.1911) - 1 ≈ 0.211)

**Key Insights**:
1. **Bathrooms have the strongest marginal effect** on price (72.5% per bathroom), followed by bedrooms (43.2%)
2. **Accommodates shows moderate effect** (15.9% per guest), indicating capacity is valued but with diminishing returns
3. **Beds show weakest effect** (21.1% per bed), likely because beds are more flexible than fixed rooms
4. **All relationships are statistically significant** (p < 0.001) with moderate-to-strong correlations (r = 0.57-0.64)
5. **R² values (32-41%)** indicate size attributes explain a substantial portion of price variation, but other factors (location, quality, amenities) also matter significantly

**Methodology Notes**:
- Outlier filtering: Removed listings with bathrooms > 8, bedrooms > 10, beds > 20 to focus on typical residential properties
- Log transformation: Used log(price) as dependent variable to enable percentage-based interpretation
- All regression results saved to `{city}/exploration_output/{city}_linear_regression_results.csv`
- Linear fit lines and equations displayed on all scatter plots (both combined and individual charts)

**Property Characteristics**:
- `property_type`: Self-selected type (House, Apartment, Condo, etc.)
- `room_type`: Entire home/apt, Private room, Shared room, Hotel room
  - **NOTE**: Hotel rooms are excluded from analysis (commercial operations, not residential investments)
- `neighbourhood_cleansed`: Geocoded neighborhood
- `zipcode`: Postal code (when available)

### Data Quality Issues Identified

1. **Occupancy Capping**: 255-day cap on `estimated_occupancy_l365d` affects ~7% of listings
2. **Negative Host-Blocked Days**: 26.65% of listings have negative host-blocked days (when `availability_365 + estimated_occupancy_l365d > 365`). This suggests the two metrics may be calculated over different time periods or using different methodologies. See `DATA_QUALITY_NOTES.md` for details.
3. **Missing Price Data**: ~30% of listings lack price information
4. **Host-Blocked Days**: Calendar shows unavailability but doesn't distinguish booked vs. host-blocked
5. **Metro-Level Prices**: Zillow data is metro-level, not property-specific

---

## 3. Data Transformations

### Priority 1: Log Transformations

**Rationale**: Investors think in percentages (returns), not absolute dollars. Log differences = percentage changes, enabling elasticity interpretation.

**Transformations Applied**:
- `log_price`: Natural log of `price_clean`
- `log_price_per_accommodates`: Log of price per guest capacity (size-adjusted pricing)
- `log_reviews`: Log of (number_of_reviews + 1) to handle zeros
- `log_accommodates`: Log of guest capacity
- `log_beds`: Log of (beds + 1) to handle zeros

**Impact**: Normalizes right-skewed distributions, enables percentage-based analysis

### Priority 2: Within-City Standardization

**Rationale**: Enables fair comparison within each city market, accounting for city-specific price levels.

**Metrics Created**:
- `price_zscore`: Standardized price (mean=0, std=1) within city
- `price_percentile`: Price percentile (0=cheapest, 1=most expensive) within city
- `log_price_zscore`: Standardized log price (for relative % differences)
- `reviews_zscore`: Standardized review count within city
- `reviews_percentile`: Review count percentile (visibility ranking)
- `reviews_per_month_zscore`: Standardized review velocity

**Impact**: Removes city-level price differences, focuses on relative positioning within market

### Priority 3: Revenue Proxies

**Rationale**: Investors care about RETURNS, not just prices. Revenue = Price × Occupancy × Days.

**PRIMARY Metrics (Calendar-Based Occupancy)**:
- `occupancy_rate`: (365 - availability_365) / 365
  - Includes booked days + host-blocked days
  - Captures full market capacity
- `est_annual_revenue`: `price_clean` × (365 - availability_365)
  - Revenue using calendar-based occupancy
- `revenue_per_accommodates`: Revenue per guest capacity (unit economics)
- `revenue_per_bedroom`: Revenue per bedroom
- `log_est_revenue`: Log-transformed revenue

**SECONDARY Metrics (Booked Days Only)**:
- `occupancy_rate_booked`: `estimated_occupancy_l365d` / 365
  - Only actual booked days (excludes host-blocked)
  - Capped at 69.9% (255 days)
- `est_annual_revenue_booked`: `price_clean` × `estimated_occupancy_l365d`
  - Revenue using only booked days

**Impact**: Enables revenue-based property comparison and ROI calculations

### Priority 4: Professionalization Metrics

**Rationale**: Market structure affects competition and pricing dynamics.

**Host-Level Metrics**:
- `host_is_professional`: Host has 2+ listings (binary)
- `host_listings_in_city`: Number of listings per host in city
- `host_professional_tier`: 
  - casual (1 listing)
  - small_professional (2-5 listings)
  - medium_professional (6-20 listings)
  - large_professional (21+ listings)

**Market-Level Metrics**:
- `pct_professional_hosts`: % of hosts with 2+ listings
- `pct_large_operators`: % of listings from hosts with 21+ properties
- `median_host_listings`: Median listings per host
- `gini_host_concentration`: Gini coefficient of host concentration (0=equal, 1=monopoly)
- `market_professionalization_score`: Composite score (0-100) combining all metrics

**Impact**: Identifies markets with different competitive structures

### Zillow Price Integration

**Rationale**: Need actual purchase prices to calculate real ROI, not just revenue proxies.

**Process**:
1. Load Zillow ZHVI, ZORI, and monthly payment data
2. Match Airbnb cities to Zillow metro areas
3. Assign metro-level prices to all listings in city

**Metrics Added**:
- `purchase_price`: Metro-level home value (ZHVI)
- `monthly_payment`: Monthly mortgage payment (20% down, middle tier)
- `zori_rent`: Metro-level rental index
- `zillow_metro`: Matched Zillow metro area name

**Limitations**:
- Metro-level prices (not property-specific)
- Some cities unmatchable (e.g., Paris - no Zillow US data)
- Assumes all properties in metro have same purchase price

---

## 4. Computed Metrics

### ROI Metrics (PRIMARY - Calendar-Based)

**Annual Cash Flow**:
```
annual_cash_flow = est_annual_revenue - (monthly_payment × 12)
```

**Cash-on-Cash ROI**:
```
cash_on_cash_roi = (annual_cash_flow / (purchase_price × 0.20)) × 100
```
- Assumes 20% downpayment
- Measures return on invested capital

**Cap Rate**:
```
cap_rate = (est_annual_revenue / purchase_price) × 100
```
- Gross cap rate (before expenses)
- Measures yield on purchase price

**Price-to-Rent Ratio**:
```
price_to_rent_ratio = purchase_price / (est_annual_revenue / 12)
```
- Months to pay off property at current revenue
- Lower = better

**Revenue Yield**:
```
revenue_yield = (est_annual_revenue / purchase_price) × 100
```
- Same as cap rate, different interpretation

### ROI Metrics (SECONDARY - Booked Days Only)

Same calculations as above, but using:
- `est_annual_revenue_booked` instead of `est_annual_revenue`
- `occupancy_rate_booked` instead of `occupancy_rate`

**Naming Convention**: All secondary metrics have `_booked` suffix
- `annual_cash_flow_booked`
- `cash_on_cash_roi_booked`
- `cap_rate_booked`
- `revenue_yield_booked`

---

## 5. Occupancy Metric Rationale

### Decision: Use Calendar-Based Occupancy as PRIMARY

**We use `occupancy_rate` (calendar-based) as PRIMARY and `occupancy_rate_booked` (booked days) as SECONDARY.**

### Rationale for Calendar-Based Occupancy

#### 1. Opportunity Cost / Economic Utility Argument
When hosts block days, they implicitly value those days at least as much as the revenue they forego. If a host blocks 30 days at $200/night, they're signaling those days are worth ≥$6,000 to them (personal use, maintenance, strategic pricing, etc.). From an investor's perspective, this represents potential revenue that could be captured if the property were managed differently. The opportunity cost equals the foregone revenue, so these days should be counted in the "theoretical utility" calculation.

#### 2. Market Capacity / True Demand Argument
Calendar unavailability (booked + blocked) reflects total market demand. Whether unavailable due to booking or blocking, the property was in demand. Blocked days may reflect:
- Strategic pricing (holding for peak season)
- Personal use (host values it highly)
- Maintenance/upgrades (preserving asset value)

This shows the property's total market capacity, not just realized bookings.

#### 3. Investment Valuation / Maximum Potential Revenue Argument
For ROI analysis, we want maximum potential revenue. Host-blocked days represent revenue that could be captured with different management. A property with 80% calendar unavailability (60% booked + 20% blocked) has higher potential than one with 60% booked and 40% available. The 20% blocked days are a management choice, not a market limitation.

#### 4. Distribution Completeness / Statistical Validity Argument
The capped metric (`estimated_occupancy_l365d`) truncates the distribution and introduces systematic downward bias:
- The 255-day cap (69.9%) creates an artificial ceiling
- Many high-performing properties are compressed into the 0.7 bucket
- This skews medians, means, and correlations downward
- The calendar proxy provides the full range of market behavior

**Supporting Evidence**: Analysis showed the "missing mass" above 0.7 is 3x larger than the capped listings, indicating significant information loss.

#### 5. Behavioral Signal / Market Quality Argument
High host-blocking rates signal high property value. Hosts who block many days are signaling:
- The property is valuable enough to forego rental income
- They have alternative uses (personal, family, etc.)
- They can afford to be selective (market power)

This is a positive signal about property desirability, not a negative signal about market demand.

#### 6. Comparability / Fair Comparison Argument
Calendar proxy enables fair comparison across all properties:
- Some properties are capped at 0.7 but may actually have 0.85+ occupancy
- Using the capped metric makes high performers look identical to moderate performers
- Calendar proxy distinguishes between truly high-occupancy properties and those artificially capped

#### 7. Practical Investment Decision Argument
Investors can influence host-blocking behavior through management:
- If you buy a property, you control whether to block days
- The calendar proxy shows what the property could earn with different management
- This is more relevant for investment decisions than what the previous host chose to do

ROI should reflect potential under optimal management, not just historical realized performance.

### Why Keep Both Metrics?

We maintain both metrics separately (not combined) to:
1. **Enable Comparison**: Show profitability under both scenarios
2. **Transparency**: Acknowledge the difference between booked days and total unavailability
3. **Sensitivity Analysis**: Demonstrate robustness of findings
4. **Documentation**: Flag capped listings with `occupancy_is_capped` flag

---

## 6. Profitability Analysis

### Individual Property Level

**Segmentation Approach**:
- Segment by: `room_type` × `size_bin` (Studio, 1BR, 2BR, 3+BR)
- Calculate median ROI metrics for each segment
- Rank segments by cash-on-cash ROI

**Key Metrics per Segment**:
- Median cash-on-cash ROI (PRIMARY and SECONDARY)
- Median cap rate
- Median revenue yield
- Median occupancy rate
- Number of listings (market size)
- Percentage of market

**Output**: `{city}_property_segments.csv` with ranked segments

### Geographic Analysis (Neighborhood Level)

**Aggregation Units** (in order of preference):
1. `zipcode` (if available and complete)
2. `neighbourhood_cleansed` (geocoded neighborhoods)
3. `neighbourhood_group_cleansed` (larger areas)

**Metrics per Geographic Unit**:
- Median ROI metrics (cash-on-cash ROI, cap rate)
- Median revenue yield
- Listing density
- Identification of "hotspots": neighborhoods with high ROI + high occupancy

**Output**: Geographic ROI heatmaps (planned)

### Property Type Analysis

**Segments Analyzed**:
- Entire home/apt × Studio
- Entire home/apt × 1BR
- Entire home/apt × 2BR
- Entire home/apt × 3+BR
- Private room × [size bins]
- Shared room × [size bins]

**Excluded**: Hotel rooms (commercial operations, not residential rental investments)

**Findings** (example from Austin):
- Best segment: Entire home/apt × 3+BR (median ROI: -25.6% PRIMARY, lower SECONDARY)
- Worst segment: Private room × Studio (median ROI: -42.6%)
- **Note**: Negative ROI indicates Airbnb revenue alone doesn't cover mortgage payments at current prices

---

## 7. City-Level Analysis

### City Comparison Metrics

**Market Structure**:
- Total listings
- Professionalization score
- % Professional hosts
- Gini coefficient of host concentration

**Pricing**:
- Median price
- Price range
- Price per accommodates

**Occupancy**:
- Median occupancy rate (PRIMARY)
- Median occupancy rate (SECONDARY)
- % of listings capped at 255 days

**Revenue**:
- Median annual revenue (PRIMARY)
- Median annual revenue (SECONDARY)
- Revenue per accommodates

**ROI**:
- Median cash-on-cash ROI (PRIMARY)
- Median cash-on-cash ROI (SECONDARY)
- Median cap rate
- % of properties with positive cash flow

**Zillow Integration**:
- Purchase price (metro-level)
- Monthly payment
- ZORI rent index

### City-Level Findings

**Example: Austin, TX**
- Total listings: 15,187
- Purchase price: $426,454 (metro-level)
- Monthly payment: $3,028
- Median cap rate (PRIMARY): 1.66%
- Median cash-on-cash ROI (PRIMARY): -34.3%
- Median cash-on-cash ROI (SECONDARY): Lower (more negative)
- Professionalization score: 39.6/100
- % Professional hosts: 53.7%

**Key Insight**: Negative ROI suggests either:
- Properties are overpriced relative to Airbnb revenue potential
- Additional income streams or appreciation needed for profitability
- Different financing terms might change results

---

## 8. Key Findings

### Occupancy Analysis

1. **Calendar-based occupancy shows higher values** than booked-days-only metric
2. **Upper end of occupancy range (70%+) appears to be the profitable band**
3. **Capped metric creates downward bias** in profitability analysis
4. **Host-blocked days represent significant potential revenue** (10-20% of calendar unavailability)

### ROI Analysis

1. **Most properties show negative cash-on-cash ROI** at current prices
   - Median ROI (PRIMARY): -34.3% (Austin example)
   - Median ROI (SECONDARY): Even more negative
2. **Larger properties (3+BR) perform better** than smaller units
3. **Entire homes outperform private/shared rooms** in ROI
4. **Calendar-based occupancy reveals more profitable opportunities** than booked-days-only analysis

### Market Structure

1. **Professionalization varies significantly** across cities
   - Range: ~20-60% professional hosts
   - Gini coefficients: 0.6-0.8 (high concentration)
2. **Professional hosts tend to have higher occupancy** (better management)
3. **Market concentration affects pricing dynamics**

### Geographic Patterns

1. **Within-city variation is significant** (neighborhood-level analysis needed)
2. **ROI hotspots exist** even in cities with overall negative ROI
3. **Property type preferences vary by geography**

---

## 9. Visualization Suite

### ROI Visualization Charts (4-Panel)

**Chart 1: ROI Distribution by Room Type**
- Violin plots showing distribution of cash-on-cash ROI
- Mean (dashed line) and Median (solid line) shown
- Focused on center mass (outliers clipped for visualization)

**Chart 2: ROI Distribution by Property Size**
- Box plots by bedroom count (Studio, 1BR, 2BR, 3+BR)
- Box = IQR, Line = Median, Dashed = Mean
- Sample sizes shown

**Chart 3: Top 10 Segments by ROI**
- Horizontal bar chart ranking segments
- Only segments with ≥10 listings
- Color-coded by ROI performance
- Sample sizes shown

**Chart 4: ROI vs Occupancy Rate**
- Scatter plot colored by room type
- Correlation coefficient displayed
- Trend line shown
- Focused on center mass

### Dual-Metric Support

**Both PRIMARY and SECONDARY visualizations generated**:
- Files suffixed with `_primary` or `_secondary`
- Combined 4-panel visualization in `analysis_output/`
- Individual charts in `exploration_output/roi_individual_charts/`

**Usage**:
```bash
# Generate PRIMARY (calendar-based) visualizations
python visualize_roi_results.py Austin -all

# Generate SECONDARY (booked days) visualizations
python visualize_roi_results.py Austin -all -secondary

# Generate BOTH versions
python visualize_roi_results.py Austin -all -both
```

---

## 10. Methodology Notes

### Data Quality Considerations

1. **Metro-Level Prices**: Zillow data is metro-level, not property-specific. All listings in a metro get the same purchase price.

2. **Property Type Mismatch**: Zillow ZHVI is for single-family homes and condos, but Airbnb includes many property types (apartments, rooms, etc.).

3. **Unmatchable Cities**: Some Airbnb cities (e.g., "Paris") have no Zillow data.

4. **Sub-City Matching**: Some Airbnb cities are sub-cities within metros (e.g., "Jersey_City" uses "New York, NY" metro prices).

5. **Time Period**: All data from 2025 (June-October), no inflation adjustment needed for cross-city comparisons.

### Assumptions

1. **Downpayment**: 20% assumed for cash-on-cash ROI calculations
2. **Monthly Payment**: Uses Zillow's calculated monthly payment (includes P&I, taxes, insurance)
3. **Revenue Calculation**: Assumes price × occupancy = revenue (simplified, doesn't account for cleaning fees, service fees, etc.)
4. **Occupancy Stability**: Assumes current occupancy rates are representative of future performance

### Limitations

1. **No Operating Expenses**: ROI calculations don't include:
   - Property management fees
   - Maintenance and repairs
   - Utilities
   - Insurance (beyond what's in monthly payment)
   - Property taxes (beyond what's in monthly payment)

2. **No Seasonality**: Analysis uses annual averages, doesn't account for seasonal variation

3. **No Regulatory Risk**: Doesn't account for potential short-term rental regulations

4. **No Appreciation**: Focus is on cash flow, not long-term appreciation

5. **Simplified Revenue Model**: Doesn't account for:
   - Dynamic pricing
   - Minimum stay requirements
   - Cleaning fees
   - Airbnb service fees

### Future Enhancements

1. **Neighborhood-Level ROI Analysis**: Geographic heatmaps showing ROI by zipcode/neighborhood
2. **Review Velocity Metrics**: Growth indicators from review data
3. **Seasonality Analysis**: Monthly/quarterly occupancy patterns
4. **Operating Expense Estimates**: More complete ROI calculations
5. **Sensitivity Analysis**: Monte Carlo simulations for risk assessment

---

## Appendix: File Structure

### Key Scripts

- `city_level_analysis.py`: Core feature engineering functions
- `visualize_roi_results.py`: ROI visualization generation
- `analyze_property_segments.py`: Property type × size segmentation
- `load_zillow_data.py`: Zillow data loading
- `match_cities_to_zillow.py`: City-to-metro matching

### Output Files

**Per City**:
- `{city}/analysis_output/{city}_roi_visualizations_primary.png`: Combined 4-panel (PRIMARY)
- `{city}/analysis_output/{city}_roi_visualizations_secondary.png`: Combined 4-panel (SECONDARY)
- `{city}/analysis_output/{city}_property_segments.csv`: Segment rankings
- `{city}/exploration_output/roi_individual_charts/`: Individual chart images

**Cross-City**:
- `city_comparison_outputs/property_segments_all_cities.csv`: Combined segment analysis

---

## Contact and Questions

For questions about methodology, data sources, or analysis approach, refer to:
- Code documentation in `city_level_analysis.py`
- Term dictionary: `term dictionary.csv`
- Work plan: `work_plan.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Status**: Complete - Ready for Presentation Development

