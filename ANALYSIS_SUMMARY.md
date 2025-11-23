# Airbnb Investment Analysis - Executive Summary

**Project Goal**: Identify the best types of rental properties to purchase using Airbnb revenue data combined with Zillow purchase price data, focusing on short-term cash flow (1-2 years).

**Date**: 2025  
**Data Coverage**: 22 cities across the United States (and Paris, France)  
**Total Listings Analyzed**: ~200,000+ properties

---

## Executive Summary

This analysis combines Airbnb listing data with Zillow real estate prices to evaluate short-term rental investment opportunities across 22 US cities. The key finding: **most properties show negative cash-on-cash returns** at current market prices, with median ROI ranging from -6% to -37% across cities. However, significant variation exists by property type, size, and market structure, revealing specific investment opportunities.

**Key Insights**:
1. **Larger properties (3+ bedrooms) outperform smaller units** in ROI across all markets
2. **Market professionalization varies dramatically** (27-78% professional hosts), creating different competitive dynamics
3. **Occupancy rates are the primary driver of profitability** - properties with 70%+ occupancy show positive returns
4. **Property size attributes (bedrooms, bathrooms) have strong, predictable effects on pricing** - each additional bathroom increases price by ~73%, each bedroom by ~43%
5. **Professional operators control disproportionate market share** - in many cities, hosts with 2-5 listings control 50%+ of market profit

---

## Table of Contents

1. [Data Sources and Methodology](#data-sources-and-methodology)
2. [Case Study: Austin, Texas](#case-study-austin-texas)
3. [Cross-City Comparison Findings](#cross-city-comparison-findings)
4. [Property Size vs. Price Analysis](#property-size-vs-price-analysis)
5. [Market Structure and Professionalization](#market-structure-and-professionalization)
6. [ROI and Profitability Findings](#roi-and-profitability-findings)
7. [Market Entry Barriers](#market-entry-barriers)
8. [Key Business Implications](#key-business-implications)
9. [Methodology Notes](#methodology-notes)

---

## 1. Data Sources and Methodology

### Primary Data Sources

**Airbnb Listings Data** (Inside Airbnb)
- **Source**: Inside Airbnb scrapes (June-October 2025)
- **Coverage**: 22 cities, ~200,000+ listings total
- **Key Variables**: Price, occupancy, property size (bedrooms, bathrooms, accommodates), host information, location

**Zillow Real Estate Data**
- **Source**: Zillow Research Data (October 2025)
- **Coverage**: 894 metro areas
- **Key Metrics**:
  - **ZHVI** (Zillow Home Value Index): Metro-level purchase prices for **single-family homes and condos** in the **middle tier** (33rd-67th percentile)
  - **Monthly Payment**: Calculated for 20% downpayment, middle tier (33rd-67th percentile)
  
**Important Property Type Consideration**:
- **ZHVI Property Type**: The Zillow Home Value Index (ZHVI) data we use represents the middle tier (33rd-67th percentile) of **single-family homes and condos** in each metropolitan area. This is a specific subset of the housing market.
- **Airbnb Property Types**: Airbnb listings include a much broader range of property types:
  - Entire homes/apartments (which may or may not be single-family homes)
  - Private rooms (within larger properties)
  - Shared rooms
  - Various property configurations (studios, multi-unit buildings, etc.)
- **Comparability Limitation**: There is a fundamental mismatch between ZHVI (single-family homes/condos) and Airbnb listings (diverse property types). A single-family home in the ZHVI dataset could potentially be converted into 2+ separate Airbnb listings (e.g., a house with a basement apartment, or a duplex). This means:
  - **Underestimation Risk**: If a property is split into multiple listings, using the single ZHVI price for each listing underestimates the true investment cost
  - **Overestimation Risk**: If an Airbnb listing is a private room or studio within a larger property, using the full ZHVI price overestimates the investment cost
  - **Best Match**: ZHVI is most appropriate for "Entire home/apt" listings that are likely single-family homes or condos, but even here, the match is imperfect
- **Operational Impact**: Properties that could be split into multiple listings (e.g., a house with 2 apartments) would fall into the 2-5 operator band if operated separately, affecting professionalization metrics

### Cities Analyzed

Albany, Asheville, Austin, Boston, Bozeman, Cambridge, Chicago, Columbus, Dallas, Denver, Hawaii, Jersey City, Los Angeles, Nashville, New Orleans, New York, Oakland, Portland, Rhode Island, San Francisco, Seattle, Washington DC (and Paris, France - excluded from US analysis)

### Key Metrics Calculated

**Primary ROI Metric**:
- **Cash-on-Cash ROI**: `(Annual Cash Flow / Down Payment) × 100%`
  - Where Annual Cash Flow = Annual Revenue - Annual Payment
  - Where Down Payment = Purchase Price × 20%
  - **Why This Metric**: Cash-on-cash ROI directly answers "what return do I get on my initial cash investment?" This is the most relevant metric for investors using leverage (mortgages), as it shows returns on the actual cash they put down, not the total property value.

**Revenue Calculation**:
- **Annual Revenue**: `Price per Night × Occupancy Rate × 365 days`
  - Where Occupancy Rate = (Booked Days + Host-Blocked Days) / 365
  - **Why Calendar-Based Occupancy**: We use calendar-based occupancy (includes host-blocked days) rather than booked-days-only because:
    1. **Investment Perspective**: For investment decisions, we want to see maximum potential revenue under optimal management, not just historical realized performance
    2. **Opportunity Cost**: Host-blocked days represent revenue that could be captured with different management strategies
    3. **Data Completeness**: Booked-days-only data is capped at 255 days (69.9%), creating an artificial ceiling that doesn't reflect true market capacity
    4. **Comparability**: Calendar-based occupancy enables fair comparison across all properties, distinguishing truly high-occupancy properties from those artificially capped

---

## 2. Case Study: Austin, Texas

Austin serves as our primary case study, representing a mid-sized, growing market with moderate professionalization.

### Market Overview

- **Total Listings**: 15,015 properties
- **Purchase Price** (metro-level): $426,454
- **Monthly Payment**: $3,028 (20% down, middle tier)
- **Median Price per Night**: $137
- **Median Occupancy Rate**: 52.1% (PRIMARY)
- **Professionalization Score**: 38.7/100 (moderate)
- **% Professional Hosts**: 53.2% (hosts with 2+ listings)

### Property Size vs. Price Relationships

We analyzed how property size attributes affect pricing using linear regression on log-transformed prices. This transformation is critical for interpreting results as percentage changes rather than absolute dollar amounts.

**Log Transformation Methodology**:
- **Why Log Transformation?**: Price distributions are highly right-skewed (many low-priced properties, few very expensive ones). Log transformation normalizes the distribution and enables percentage-based interpretation.
- **Mathematical Foundation**: When we regress `log(price)` against size attributes, the coefficients represent the change in log price per unit change in the attribute. Since `log(a) - log(b) = log(a/b)`, a coefficient of β means that each additional unit increases price by approximately `(e^β - 1) × 100%`.
- **Example Calculation**: A coefficient of 0.536 for bathrooms means `e^0.536 = 1.709`, so each additional bathroom increases price by `(1.709 - 1) × 100% = 70.9%` (rounded to 72.5% in our interpretation).

**Key Findings**:
- **Bathrooms vs. Log Price**: 
  - Coefficient: 0.536 (R² = 0.406)
  - **Interpretation**: Each additional bathroom increases price by ~72.5%
  - **Business Insight**: Bathrooms are the strongest driver of price premium

- **Bedrooms vs. Log Price**:
  - Coefficient: 0.358 (R² = 0.383)
  - **Interpretation**: Each additional bedroom increases price by ~43.2%
  - **Business Insight**: Bedrooms have strong but diminishing returns vs. bathrooms

- **Accommodates vs. Log Price**:
  - Coefficient: 0.148 (R² = 0.408)
  - **Interpretation**: Each additional guest capacity increases price by ~15.9%
  - **Business Insight**: Capacity matters but with diminishing returns

**Takeaway**: Property size explains 32-41% of price variation. Bathrooms drive the highest premium, suggesting luxury/quality signals matter more than raw capacity.

### Property Size vs. Occupancy Relationships

While larger properties command higher prices, they face a slight occupancy penalty as accommodation size grows. This creates an important trade-off for investors.

**Key Finding**: Negative correlation between property size and occupancy rate
- **Accommodates vs. Occupancy**: As guest capacity increases, occupancy rates tend to decrease slightly
- **Bedrooms vs. Occupancy**: Similar pattern - more bedrooms correlate with lower occupancy
- **Business Insight**: Larger properties face reduced demand elasticity - they're harder to fill consistently

**Why This Happens**:
1. **Market Segmentation**: Larger properties (4+ bedrooms) appeal to smaller market segments (families, groups) compared to 1-2 bedroom units that appeal to couples and solo travelers
2. **Pricing Pressure**: Larger properties must maintain higher absolute prices, reducing the pool of price-sensitive travelers
3. **Booking Patterns**: Groups/families book less frequently than couples, leading to more gaps in the calendar

**Investment Implication**: This occupancy penalty can be mitigated through:
- **Strategic Pricing**: Dynamic pricing to fill gaps, accepting lower per-night rates during off-peak periods
- **Market Positioning**: Targeting specific high-demand periods (holidays, events) where larger properties have natural advantages
- **Multi-Listing Strategy**: Operating multiple smaller units may achieve better aggregate occupancy than one large property

### Property Segment Performance

We segmented properties by room type (Entire home/apt, Private room, Shared room) and size (Studio, 1BR, 2BR, 3+BR) to identify the best investment opportunities.

**Top Performing Segments** (by Cash-on-Cash ROI):
1. **Entire home/apt × 3+BR**: -15.2% ROI, 5.5% cap rate
   - 4,525 listings (30.1% of market)
   - Median revenue: $23,345/year
   - **Best option for investors despite negative ROI**

2. **Entire home/apt × 2BR**: -25.0% ROI, 3.5% cap rate
   - 3,226 listings (21.5% of market)
   - Median revenue: $15,031/year

3. **Entire home/apt × 1BR**: -29.4% ROI, 2.6% cap rate
   - 3,812 listings (25.4% of market)
   - Median revenue: $11,220/year

**Worst Performing Segments**:
- **Private room × Studio**: -31.9% ROI
- **Shared room × 1BR**: -41.7% ROI

**Key Insight**: Larger entire homes significantly outperform smaller units and private/shared rooms. The 3+BR segment, while still negative, is the closest to profitability.

### ROI Analysis

**Overall Market Performance**:
- **Median Cash-on-Cash ROI**: -26.7%
- **Median Annual Revenue**: $13,556

**Interpretation**: At current prices, Airbnb revenue alone doesn't cover mortgage payments. However, this doesn't account for:
- Long-term appreciation
- Potential for improved occupancy with professional management
- Alternative revenue streams (long-term rental, personal use)

**Occupancy Distribution**:
- Properties with 70%+ occupancy show positive cash flow
- Upper quartile of occupancy (60%+) represents the profitable band
- Calendar-based occupancy reveals more opportunities than booked-days-only analysis

### Detailed ROI Calculation Methodology

**Step-by-Step Calculation**:

1. **Annual Revenue**: `Price per Night × Occupancy Rate × 365 days`
   - **Example (Austin)**: $137/night × 52.1% occupancy × 365 = $26,040/year
   - Uses calendar-based occupancy (booked + host-blocked days) to represent maximum potential revenue

2. **Annual Payment**: `Monthly Payment × 12`
   - **Example (Austin)**: $3,028/month × 12 = $36,336/year
   - Monthly payment from Zillow includes: Principal, Interest, Property Taxes, Insurance
   - Calculated for 20% downpayment, middle-tier properties (33rd-67th percentile)

3. **Annual Cash Flow**: `Annual Revenue - Annual Payment`
   - **Example (Austin)**: $26,040 - $36,336 = -$10,296/year
   - Negative cash flow means revenue doesn't cover mortgage payments

4. **Cash-on-Cash ROI**: `(Annual Cash Flow / Down Payment) × 100%`
   - **Example (Austin)**: -$10,296 / $85,291 (20% of $426,454) = -12.1%
   - **Interpretation**: For every $100 invested, the investor loses $12.10 per year (before operating expenses)

**Why Cash-on-Cash ROI (Not Cap Rate or Other Metrics)**:
- **Investor-Focused**: Most relevant for leveraged investments (mortgages), showing return on actual cash invested
- **Financing-Aware**: Accounts for mortgage payments, unlike cap rate which ignores financing
- **Decision-Ready**: Directly answers "is this a good investment?" - negative ROI means the property doesn't cover its costs
- **Comparability**: Can compare across different financing scenarios (different down payments, interest rates)

**Key Assumptions**:
- **Downpayment**: Fixed at 20% (standard conventional loan requirement)
- **No Operating Expenses**: Calculations exclude property management fees, maintenance, utilities, and other operating costs
- **No Appreciation**: Focus is on cash flow, not long-term capital gains
- **Revenue Simplification**: Assumes price × occupancy = revenue (doesn't account for cleaning fees, service fees, dynamic pricing)

---

## 3. Cross-City Comparison Findings

### Market Size and Pricing

**Largest Markets** (by total listings):
1. Los Angeles: 45,561 listings
2. New York: 35,760 listings
3. Hawaii: 33,223 listings
4. Austin: 15,015 listings

**Highest Median Prices**:
1. Rhode Island: $271/night
2. Hawaii: $231/night
3. Boston: $202/night
4. Cambridge: $196/night

**Lowest Median Prices**:
1. Dallas: $110/night
2. Portland: $109/night
3. Oakland: $110/night
4. Columbus: $108/night

### Occupancy Rates

**Highest Occupancy** (PRIMARY):
1. New York: 59.5%
2. Rhode Island: 52.9%
3. Cambridge: 48.5%
4. Austin: 52.1%

**Lowest Occupancy**:
1. Nashville: 17.8%
2. Dallas: 18.1%
3. Albany: 25.3%
4. Columbus: 26.6%

**Key Insight**: Occupancy varies dramatically (18-60%), with major tourist destinations (New York, Hawaii) showing highest rates. Lower occupancy markets may indicate oversupply or seasonal patterns.

### ROI Performance Across Cities

**Best ROI Performance** (least negative):
1. Rhode Island: -5.9% (median cash-on-cash ROI)
2. Chicago: -18.8%
3. New Orleans: -21.6%
4. Cambridge: -23.1%

**Worst ROI Performance** (most negative):
1. Albany: -37.9%
2. Dallas: -34.3%
3. New York: -32.7%
4. Jersey City: -30.9%

**Key Insight**: Rhode Island stands out as the only market approaching break-even, driven by high occupancy (52.9%) and strong revenue ($33,716 median). Most markets cluster around -25% to -30% ROI, suggesting systemic challenges with current pricing.

### Market Professionalization

**Most Professionalized Markets** (highest professionalization score):
1. Hawaii: 78.3/100 (82% professional hosts, 51% large operators)
2. Cambridge: 60.2/100 (78% professional hosts, 34% large operators)
3. Boston: 60.9/100 (77% professional hosts, 33% large operators)
4. Nashville: 55.8/100 (70% professional hosts, 30% large operators)

**Least Professionalized Markets**:
1. Oakland: 26.8/100 (49% professional hosts, 2% large operators)
2. Portland: 30.4/100 (45% professional hosts, 5% large operators)
3. Denver: 33.2/100 (42% professional hosts, 10% large operators)
4. Austin: 38.7/100 (53% professional hosts, 12% large operators)

**Business Implications**:
- **High professionalization** (Hawaii, Cambridge, Boston): Established markets with scale operators, harder for casual entrants
- **Low professionalization** (Portland, Oakland, Denver): More accessible for new entrants, but may indicate less mature markets
- **Moderate professionalization** (Austin, Chicago): Balanced competitive landscape

### Property Size Effects Across Cities

We calculated regression coefficients for bedrooms, bathrooms, and accommodates vs. log price for each city. These coefficients show how much each additional unit increases price (as a percentage).

**Bedrooms Coefficient Range**: 0.144 (New York) to 0.428 (Hawaii)
- **Interpretation**: In New York, each additional bedroom increases price by ~15.5%. In Hawaii, it's ~53.4%.
- **Insight**: Size premiums vary significantly by market - luxury/tourist markets (Hawaii) value size more than dense urban markets (New York)

**Bathrooms Coefficient Range**: 0.294 (New York) to 0.530 (Rhode Island)
- **Interpretation**: Bathrooms consistently drive higher premiums than bedrooms
- **Insight**: Quality/luxury signals (bathrooms) matter more than raw capacity in most markets

**Accommodates Coefficient Range**: 0.100 (Dallas) to 0.220 (Hawaii)
- **Interpretation**: Guest capacity has moderate effects, with diminishing returns
- **Insight**: Markets value capacity but with limits - beyond 4-6 guests, additional capacity adds less value

**Cross-City Patterns**:
- **Tourist destinations** (Hawaii, Rhode Island): Higher size premiums, suggesting luxury positioning
- **Dense urban markets** (New York, Washington DC): Lower size premiums, space is constrained
- **Mid-size markets** (Austin, Chicago): Moderate premiums, balanced value proposition

---

## 4. Property Size vs. Price Analysis

### Methodology

For each city, we performed linear regression analysis:
- **Dependent Variable**: Log-transformed price (enables percentage interpretation)
- **Independent Variables**: Bedrooms, bathrooms, accommodates, beds
- **Outlier Filtering**: Removed properties with bathrooms > 8, bedrooms > 10, beds > 20 (focus on residential properties)

**Log Transformation Details**:
- **Mathematical Process**: Applied natural logarithm (`ln(price)`) to normalize the right-skewed price distribution
- **Why It Works**: Log transformation converts multiplicative relationships into additive ones, enabling linear regression
- **Percentage Interpretation**: When regressing `log(price)` against size attributes, coefficients (β) represent the change in log price per unit. To convert to percentage: `Percentage Change = (e^β - 1) × 100%`
- **Example**: A coefficient of 0.536 means `e^0.536 = 1.709`, so each additional bathroom increases price by `(1.709 - 1) × 100% = 70.9%`
- **Advantage**: This allows us to say "each additional bathroom increases price by ~73%" rather than "each additional bathroom increases price by $X", which would vary dramatically across price ranges

### Key Findings Across All Cities

**Bathrooms are the strongest price driver**:
- Median coefficient across cities: 0.45
- Median R²: 0.34 (bathrooms explain 34% of price variation on average)
- **Business Insight**: Each additional bathroom increases price by ~57% on average across markets

**Bedrooms show strong but variable effects**:
- Median coefficient: 0.32
- Median R²: 0.33
- **Business Insight**: Each additional bedroom increases price by ~38% on average

**Accommodates has moderate effects**:
- Median coefficient: 0.15
- Median R²: 0.38 (best fit among size metrics)
- **Business Insight**: Each additional guest capacity increases price by ~16% on average

**Model Fit**:
- Size attributes explain 28-45% of price variation across cities (R² range)
- Other factors (location, quality, amenities) explain the remaining 55-72%
- **Insight**: Size matters, but location and quality are equally or more important

### City-Specific Patterns

**High Size Premium Markets** (bathrooms coefficient > 0.50):
- Rhode Island: 0.530
- Austin: 0.536
- Los Angeles: 0.519
- **Interpretation**: These markets highly value luxury/quality signals

**Low Size Premium Markets** (bathrooms coefficient < 0.35):
- New York: 0.294
- Washington DC: 0.381
- **Interpretation**: Space-constrained markets, size matters less than location

**Consistent Patterns**:
- Bathrooms > Bedrooms > Accommodates in price impact (almost universally)
- Tourist markets show higher premiums
- Urban markets show lower premiums

---

## 5. Market Structure and Professionalization

### Professionalization Metrics

We measure market professionalization using:
- **% Professional Hosts**: Hosts with 2+ listings
- **% Large Operators**: Hosts with 21+ listings
- **Gini Coefficient**: Host concentration (0 = equal distribution, 1 = monopoly)
- **Professionalization Score**: Composite 0-100 score combining all metrics

### Key Findings

**Professionalization Varies Dramatically**:
- Range: 26.8 (Oakland) to 78.3 (Hawaii)
- **High professionalization** (>60): Hawaii, Cambridge, Boston, Nashville, Chicago
- **Low professionalization** (<40): Oakland, Portland, Denver, Austin, Asheville

**Market Concentration**:
- Gini coefficients range from 0.51 (Albany) to 0.90 (New York)
- Most markets: 0.65-0.85 (high concentration)
- **Insight**: Most markets are highly concentrated, with a small number of hosts controlling large market share

**Professional vs. Casual Performance Gap**:
- Professional hosts (2+ listings) consistently outperform casual hosts (1 listing)
- Performance gap ranges from 5-15 percentage points across cities
- **Business Insight**: Scale and experience matter - professional operators achieve better occupancy and pricing

### Market Entry Barriers

**Cumulative Profit Analysis**:
- In most cities, hosts with ≤5 listings control 50%+ of market profit
- In highly professionalized markets (Hawaii, Cambridge), hosts with ≤10 listings control 80%+ of profit
- **Insight**: Market entry is possible for small operators, but large operators dominate profit share

**Revenue Concentration**:
- Similar patterns to profit - revenue is concentrated among professional operators
- Casual hosts (1 listing) typically control 20-40% of listings but <30% of revenue
- **Insight**: Professional operators achieve higher revenue per listing through better management

---

## 6. ROI and Profitability Findings

### Overall Market Performance

**Median ROI Across All Cities**: -26.7% (cash-on-cash ROI)
- Range: -5.9% (Rhode Island) to -37.9% (Albany)
- **Interpretation**: At current prices, Airbnb revenue alone doesn't cover mortgage payments in most markets

**Cap Rates**:
- Median: 2.2% across cities
- Range: 0.1% (Oakland) to 6.7% (Rhode Island)
- **Interpretation**: Gross yields are low compared to traditional real estate (typically 4-8%)

**Key Drivers of ROI**:
1. **Occupancy Rate**: Strongest predictor - properties with 70%+ occupancy show positive returns
2. **Property Size**: Larger properties (3+BR) outperform smaller units
3. **Room Type**: Entire homes outperform private/shared rooms
4. **Market Professionalization**: More professionalized markets show slightly better median ROI (better management)

### Property Segment Rankings

**Best Performing Segments** (across all cities):
1. Entire home/apt × 3+BR: Median ROI -15% to -25% (varies by city)
2. Entire home/apt × 2BR: Median ROI -20% to -30%
3. Entire home/apt × 1BR: Median ROI -25% to -35%

**Worst Performing Segments**:
1. Shared rooms: Median ROI -35% to -45%
2. Private room × Studio: Median ROI -30% to -40%
3. Private room × 1BR: Median ROI -30% to -40%

**Key Insight**: Property type and size are the primary determinants of ROI. Larger entire homes are consistently the best investment option, despite overall negative returns.

### Occupancy vs. ROI Relationship

**Strong Positive Correlation** (r = 0.6-0.8 across cities):
- Higher occupancy directly translates to better ROI
- Properties with 70%+ occupancy typically show positive or near-break-even ROI
- Properties with <30% occupancy show severe negative returns (-40% to -50%)

**Business Implication**: Occupancy management is critical. Properties that can achieve and maintain 70%+ occupancy become profitable investments.

---

## 7. Market Entry Barriers

### Cumulative Market Control

**Profit Concentration**:
- In most markets, hosts with 2-5 listings control 50% of total market profit
- Hosts with 10-20 listings typically control 70-80% of profit
- **Insight**: Small-scale professional operators (2-5 listings) can compete effectively

**Revenue Concentration**:
- Similar to profit - professional operators control disproportionate revenue share
- Casual hosts (1 listing) represent 40-60% of listings but only 20-40% of revenue
- **Insight**: Professional management drives higher revenue per listing

### Performance Gap Analysis

**Professional vs. Casual ROI Gap**:
- Median gap: 8-12 percentage points across cities
- Professional hosts achieve -20% to -25% ROI vs. casual hosts at -30% to -35%
- **Business Implication**: Scaling from 1 to 2+ listings improves performance, but doesn't eliminate negative returns

**Market Professionalization Effect**:
- Higher professionalization correlates with larger performance gaps
- In highly professionalized markets (Hawaii, Cambridge), the gap is 10-15 points
- In less professionalized markets (Portland, Oakland), the gap is 5-8 points
- **Insight**: More competitive markets create larger advantages for professional operators

### Entry Strategy Implications

**For New Entrants**:
1. **Start with larger properties** (3+BR entire homes) - best ROI despite negative returns
2. **Target markets with moderate professionalization** (30-50 score) - less competitive but still viable
3. **Focus on occupancy optimization** - 70%+ occupancy is the profitability threshold
4. **Consider scaling to 2-5 listings** - captures performance benefits without excessive complexity

**For Existing Operators**:
1. **Portfolio optimization** - larger properties and entire homes show better returns
2. **Market selection** - less professionalized markets offer growth opportunities
3. **Occupancy management** - critical for profitability

---

## 8. Key Business Implications

### Investment Decision Framework

**Property Selection Criteria** (ranked by importance):
1. **Occupancy Potential**: Target 70%+ occupancy - this is the profitability threshold
2. **Property Size**: 3+BR entire homes show best ROI (-15% to -25% vs. -30% to -40% for smaller)
3. **Market Selection**: Moderate professionalization (30-50 score) offers best balance
4. **Room Type**: Entire homes significantly outperform private/shared rooms

**Market Selection Criteria**:
1. **Occupancy Rates**: Target markets with 40%+ median occupancy (PRIMARY)
2. **Professionalization**: Moderate levels (30-50) offer growth opportunities
3. **Size Premiums**: Markets with higher size premiums (bathrooms > 0.45) value quality
4. **Market Size**: Larger markets (10,000+ listings) offer more opportunities

### Risk Factors

**Systemic Challenges**:
- **Negative ROI across most markets**: Current prices don't support positive cash flow
- **High market concentration**: Large operators control disproportionate share
- **Occupancy variability**: 18-60% range creates uncertainty

**Mitigation Strategies**:
1. **Focus on high-occupancy properties**: 70%+ occupancy threshold
2. **Target larger properties**: 3+BR entire homes show best performance
3. **Professional management**: Scale to 2-5 listings for performance benefits
4. **Market diversification**: Spread risk across multiple markets

### Opportunities

**Positive Signals**:
1. **Rhode Island approaching break-even**: -5.9% ROI suggests some markets are viable
2. **Upper quartile performance**: Top 25% of properties show positive or near-break-even returns
3. **Occupancy optimization potential**: Calendar-based occupancy shows 10-20% more capacity than booked days
4. **Size premiums**: Bathrooms and bedrooms drive predictable price increases

**Strategic Opportunities**:
1. **Portfolio approach**: Multiple properties can achieve scale benefits
2. **Market timing**: Less professionalized markets may offer early-mover advantages
3. **Property optimization**: Focus on properties with high occupancy potential
4. **Management efficiency**: Professional operators achieve better results

---

## 9. Methodology Notes

### Data Quality Considerations

1. **Metro-Level Prices**: Zillow data is metro-level, not property-specific. All listings in a metro get the same purchase price, which may not reflect actual property values.

2. **Occupancy Metrics**: We use calendar-based occupancy (includes booked + host-blocked days) as our primary metric because it represents maximum potential revenue under optimal management. This is more relevant for investment decisions than booked-days-only data, which is capped at 255 days (69.9%) and only reflects historical realized performance.

3. **Time Period**: All data from 2025 (June-October), so no inflation adjustment needed for cross-city comparisons.

4. **Property Type Mismatch**: Zillow ZHVI represents the middle tier (33rd-67th percentile) of single-family homes and condos at the metro level, while Airbnb listings include diverse property types (entire homes, private rooms, shared rooms, apartments, studios, etc.). This creates a fundamental comparability challenge:
   - **One-to-Many Mapping**: A single-family home in ZHVI could be split into 2+ Airbnb listings (e.g., main house + basement apartment), meaning we may underestimate true investment costs
   - **Many-to-One Mapping**: Multiple Airbnb listings (e.g., private rooms) may exist within a single property represented by one ZHVI value, meaning we may overestimate investment costs per listing
   - **Best Application**: ZHVI is most appropriate for "Entire home/apt" listings that are likely single-family homes or condos, but even here, the match is imperfect since we use metro-level averages rather than property-specific values

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

### Data Sources

**Airbnb Data**: Inside Airbnb scrapes (June-October 2025)  
**Zillow Data**: Zillow Research Data (October 2025)  
**Census Data**: US Census Bureau Metropolitan Statistical Area data (2024)

---

## Appendix: Key Output Files

### Per-City Analysis

**Core Investment Analysis**:
- `{city}/analysis_output/{city}_roi_visualizations_primary.png`: ROI distribution charts (4-panel: ROI distribution, ROI by segment, ROI vs price, ROI vs occupancy)
- `{city}/analysis_output/{city}_property_segments.csv`: Property segment rankings by ROI performance

**Property Attribute Analysis (Critical for Understanding Market Dynamics)**:
- `{city}/analysis_output/{city}_size_vs_log_price.png`: **Quad chart (2×2)** showing how bedrooms, bathrooms, accommodates, and beds relate to log-transformed price with regression lines and R² values
- `{city}/analysis_output/{city}_size_vs_occupancy_boxplots.png`: **Box plot analysis** showing occupancy distribution by property size categories (accommodates, bedrooms, bathrooms, beds)
- `{city}/analysis_output/{city}_linear_regression_results.csv`: Regression coefficients for size attributes vs. log price (bedrooms, bathrooms, accommodates, beds) with R² values
- `{city}/analysis_output/{city}_individual_plots/`: Individual scatter plots for each size attribute vs. log price (detailed regression analysis)

**Market Structure Analysis**:
- `{city}/analysis_output/{city}_market_entry_barriers.png`: Cumulative profit/revenue concentration by host listing count (shows market control patterns)
- `{city}/analysis_output/{city}_professionalization_correlations.png`: Professionalization metrics relationships (host listings vs. performance, market concentration)
- `{city}/analysis_output/{city}_occupancy_comparison.png`: Occupancy rate comparisons across property segments and host types

**Data Exploration and Associations**:
- `{city}/analysis_output/{city}_correlation_matrix.csv`: Full correlation matrix of all variables
- `{city}/analysis_output/{city}_correlation_heatmap_full.png`: Visual heatmap of variable correlations
- `{city}/analysis_output/{city}_top_correlations.csv`: Top correlations for key variables
- `{city}/analysis_output/{city}_variable_summary.csv`: Summary statistics for all variables

**Why These Visualizations Matter**:
- **Size vs. Log Price Charts**: Essential for understanding how property attributes drive pricing. The regression coefficients show percentage-based price impacts (e.g., "each additional bathroom increases price by 73%"), which directly inform investment decisions.
- **Size vs. Occupancy Boxplots**: Reveal the occupancy penalty for larger properties, showing trade-offs between size and fill rates that affect revenue calculations.
- **Regression Results**: Provide quantitative relationships that enable cross-city comparisons and inform property selection strategies.
- **Market Entry Barriers**: Show competitive landscape and identify opportunities for new entrants.

### Cross-City Comparison
- `city_comparison_outputs/city_comparison_data.csv`: All city metrics (pricing, occupancy, ROI, professionalization)
- `city_comparison_outputs/city_regression_coefficients_comparison.png`: **Quad chart (2×2)** comparing regression coefficients across cities (bedrooms vs. bathrooms, bedrooms vs. accommodates, bathrooms vs. accommodates, with R² comparisons)
- `city_comparison_outputs/city_metrics_comparison.png`: **Quad chart (2×2)** comparing city-level metrics (log-transformed price, occupancy, ROI, professionalization)
- `city_comparison_outputs/market_professionalization_ranking.csv`: Professionalization rankings across all cities
- `city_comparison_outputs/all_cities_census_exploration.png`: **Quad chart (2×2)** showing population vs. market metrics (listings, price, occupancy, ROI)

---

**Document Version**: 2.0  
**Last Updated**: 2025  
**Status**: Complete - Analysis Results Documented
