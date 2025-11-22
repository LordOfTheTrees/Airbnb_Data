# Data Quality Notes

## Source of `estimated_occupancy_l365d`

**Answer**: `estimated_occupancy_l365d` is a **raw field from the Airbnb dataset** (Inside Airbnb scrapes). It is NOT something we created - it comes directly from the `listings.csv.gz` file.

**Definition**: Actual booked days in the last 365 days, as calculated by Airbnb from their booking data.

**Limitation**: This field is capped at 255 days (69.9% occupancy) due to an 8-bit integer limitation in the data format. This means:
- Any property with >255 booked days shows as exactly 255
- This affects ~7% of listings in Austin
- Creates artificial ceiling and downward bias in occupancy analysis

**Why we use calendar-based occupancy as PRIMARY**: The calendar-based metric (`availability_365`) captures the full range (0-100%) and includes both booked days AND host-blocked days, providing a more complete picture of market capacity.

---

## Negative Host-Blocked Rate Issue

### Problem
26.65% of listings in Austin have **negative host-blocked days**, meaning:
- `unavailable_days < booked_days`
- Or: `(365 - availability_365) < estimated_occupancy_l365d`
- Or: `availability_365 + estimated_occupancy_l365d > 365`

This is mathematically impossible if both metrics cover the same 365-day period.

### Statistics (Austin, excluding hotel rooms)
- **Total listings**: 15,015
- **Listings with negative host_blocked_days**: 4,001 (26.65%)
- **Range of negative values**: -255 to -1 days
- **Sum check** (availability_365 + estimated_occupancy_l365d):
  - Min: 366 days
  - Max: 620 days
  - Mean: 455.2 days
  - Count > 365: 4,001 (all negative cases)
  - Count > 400: 2,869 (71.7% of negative cases)

### Possible Explanations

1. **Different Time Periods**: 
   - `availability_365` might be calculated for a different 365-day window than `estimated_occupancy_l365d`
   - One might be "rolling 365 days" while the other is "calendar year"
   - Data collection timing differences

2. **Data Inconsistency**:
   - The two metrics may be calculated using different methodologies
   - `availability_365` might include future availability projections
   - `estimated_occupancy_l365d` might be based on actual completed bookings only

3. **Overlapping Bookings**:
   - If a property has multiple units or overlapping bookings, `estimated_occupancy_l365d` might count days multiple times
   - Calendar availability might be calculated differently

4. **Data Quality Issues**:
   - Missing or incorrect values in one or both fields
   - Data entry errors
   - Inconsistent data collection methods

### Impact on Analysis

**For our analysis**: This doesn't significantly impact our PRIMARY metric (calendar-based occupancy) because:
- We use `availability_365` directly: `occupancy_rate = (365 - availability_365) / 365`
- We don't rely on the difference calculation for our primary analysis
- The negative values only appear when we try to calculate `host_blocked_days = unavailable_days - booked_days`

**Recommendation**: 
- Use calendar-based occupancy (`occupancy_rate`) as PRIMARY metric (already implemented)
- Use `estimated_occupancy_l365d` as SECONDARY metric for comparison
- Be aware that the difference between them may not always represent "host-blocked days" due to data inconsistencies
- When calculating host-blocked rates, consider capping negative values at 0 or flagging them as data quality issues

---

## Hotel Rooms Exclusion

**Rationale**: Hotel rooms represent commercial hospitality operations, not residential rental investments. They have very different:
- Pricing structures
- Occupancy patterns
- ROI characteristics
- Business models

**Impact**: 
- Austin: 172 hotel rooms excluded (1.1% of total listings)
- All analyses now focus on residential rental investments only
- Filtering applied at data loading stage in `apply_all_feature_engineering()`

