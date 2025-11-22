"""
Test script for investment analysis components.

Tests:
1. Zillow data loading
2. City-to-metro matching
3. ROI calculations
4. Property segment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_zillow_loading():
    """Test Zillow data loading functions."""
    print("=" * 80)
    print("TEST 1: ZILLOW DATA LOADING")
    print("=" * 80)
    
    try:
        from load_zillow_data import load_all_zillow_data
        
        zillow_data = load_all_zillow_data()
        
        # Check structure
        assert 'RegionID' in zillow_data.columns
        assert 'RegionName' in zillow_data.columns
        assert 'zhvi_price' in zillow_data.columns
        assert len(zillow_data) > 0
        
        # Check data quality
        assert zillow_data['zhvi_price'].notna().sum() > 0
        assert zillow_data['zhvi_price'].min() > 0
        
        print("✓ Zillow data loaded successfully")
        print(f"  - {len(zillow_data):,} metros")
        print(f"  - Price range: ${zillow_data['zhvi_price'].min():,.0f} - ${zillow_data['zhvi_price'].max():,.0f}")
        print(f"  - Metros with ZHVI: {zillow_data['zhvi_price'].notna().sum():,}")
        
        return True, zillow_data
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_city_matching(zillow_data):
    """Test city-to-metro matching."""
    print("\n" + "=" * 80)
    print("TEST 2: CITY-TO-METRO MATCHING")
    print("=" * 80)
    
    try:
        from match_cities_to_zillow import match_city_to_zillow, get_city_to_zillow_mapping
        
        # Test a few cities
        test_cities = ['Austin', 'New_York', 'Chicago', 'Jersey_City', 'Paris']
        
        all_passed = True
        for city in test_cities:
            metro_data = match_city_to_zillow(city, zillow_data)
            
            if city == 'Paris':
                # Paris should be unmatchable
                if metro_data is None:
                    print(f"✓ {city}: Correctly identified as unmatchable")
                else:
                    print(f"✗ {city}: Should be unmatchable but got data")
                    all_passed = False
            else:
                # Other cities should have matches
                if metro_data is not None:
                    print(f"✓ {city}: Matched to {metro_data['RegionName']}")
                    print(f"    Price: ${metro_data['zhvi_price']:,.0f}")
                else:
                    print(f"✗ {city}: Should have match but got None")
                    all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_calculations():
    """Test ROI calculations on sample data."""
    print("\n" + "=" * 80)
    print("TEST 3: ROI CALCULATIONS")
    print("=" * 80)
    
    try:
        from city_level_analysis import add_roi_metrics
        
        # Create sample data
        n_samples = 100
        df = pd.DataFrame({
            'purchase_price': [400000] * n_samples,  # $400k property
            'monthly_payment': [2500] * n_samples,   # $2,500/month
            'est_annual_revenue': np.random.uniform(20000, 80000, n_samples)  # $20k-$80k revenue
        })
        
        # Calculate ROI
        df = add_roi_metrics(df)
        
        # Check that metrics were created
        assert 'cash_on_cash_roi' in df.columns
        assert 'cap_rate' in df.columns
        assert 'revenue_yield' in df.columns
        assert 'annual_cash_flow' in df.columns
        
        # Check calculations are reasonable
        # Cap rate should be revenue / purchase_price
        expected_cap_rate = (df['est_annual_revenue'] / df['purchase_price'] * 100).median()
        actual_cap_rate = df['cap_rate'].median()
        
        assert abs(expected_cap_rate - actual_cap_rate) < 0.01, f"Cap rate mismatch: {expected_cap_rate} vs {actual_cap_rate}"
        
        # Cash flow should be revenue - (payment * 12)
        expected_cf = (df['est_annual_revenue'] - df['monthly_payment'] * 12).median()
        actual_cf = df['annual_cash_flow'].median()
        
        assert abs(expected_cf - actual_cf) < 1, f"Cash flow mismatch: {expected_cf} vs {actual_cf}"
        
        print("✓ ROI calculations working correctly")
        print(f"  - Sample median cap rate: {actual_cap_rate:.2f}%")
        print(f"  - Sample median cash-on-cash ROI: {df['cash_on_cash_roi'].median():.1f}%")
        print(f"  - Sample median annual cash flow: ${actual_cf:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_austin_integration():
    """Test full integration on Austin data."""
    print("\n" + "=" * 80)
    print("TEST 4: FULL INTEGRATION TEST (AUSTIN)")
    print("=" * 80)
    
    try:
        from analyze_property_segments import load_city_data
        from city_level_analysis import apply_all_feature_engineering
        
        # Load Austin data
        print("Loading Austin data...")
        df = load_city_data('Austin', base_dir='.', use_detailed=True)
        print(f"  Loaded {len(df):,} listings")
        
        # Apply feature engineering
        print("Applying feature engineering...")
        df = apply_all_feature_engineering(df, 'Austin', include_zillow=True)
        
        # Check that Zillow prices were added
        assert 'purchase_price' in df.columns, "purchase_price not found"
        assert df['purchase_price'].notna().any(), "No purchase prices found"
        
        # Check that ROI metrics were calculated
        assert 'cash_on_cash_roi' in df.columns, "cash_on_cash_roi not found"
        assert 'cap_rate' in df.columns, "cap_rate not found"
        
        # Check data quality
        n_with_roi = df['cash_on_cash_roi'].notna().sum()
        pct_with_roi = n_with_roi / len(df) * 100
        
        print(f"✓ Integration test passed")
        print(f"  - Total listings: {len(df):,}")
        print(f"  - Listings with ROI: {n_with_roi:,} ({pct_with_roi:.1f}%)")
        print(f"  - Purchase price: ${df['purchase_price'].iloc[0]:,.0f}")
        print(f"  - Median cap rate: {df['cap_rate'].median():.2f}%")
        print(f"  - Median cash-on-cash ROI: {df['cash_on_cash_roi'].median():.1f}%")
        
        # Check for reasonable values
        assert df['purchase_price'].iloc[0] > 100000, "Purchase price seems too low"
        assert df['purchase_price'].iloc[0] < 2000000, "Purchase price seems too high"
        
        return True, df
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_property_segments(df):
    """Test property segment analysis."""
    print("\n" + "=" * 80)
    print("TEST 5: PROPERTY SEGMENT ANALYSIS")
    print("=" * 80)
    
    try:
        from analyze_property_segments import analyze_property_segments
        
        # Run segment analysis
        segments_df = analyze_property_segments(df, 'Austin')
        
        if segments_df is None:
            print("✗ FAILED: No segments returned")
            return False
        
        # Check structure
        assert 'room_type' in segments_df.columns
        assert 'size_bin' in segments_df.columns
        assert 'median_cash_on_cash_roi' in segments_df.columns
        assert len(segments_df) > 0
        
        # Check that segments are ranked
        assert 'roi_rank' in segments_df.columns
        assert segments_df['roi_rank'].min() == 1
        
        print(f"✓ Property segment analysis working")
        print(f"  - Found {len(segments_df)} segments")
        print(f"  - Top segment: {segments_df.iloc[0]['segment']}")
        print(f"  - Top ROI: {segments_df.iloc[0]['median_cash_on_cash_roi']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INVESTMENT ANALYSIS - TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Zillow loading
    success, zillow_data = test_zillow_loading()
    results['zillow_loading'] = success
    
    if not success:
        print("\n⚠️  Cannot continue tests without Zillow data")
        return results
    
    # Test 2: City matching
    results['city_matching'] = test_city_matching(zillow_data)
    
    # Test 3: ROI calculations
    results['roi_calculations'] = test_roi_calculations()
    
    # Test 4: Full integration
    success, df = test_austin_integration()
    results['integration'] = success
    
    if not success or df is None:
        print("\n⚠️  Cannot test property segments without integration")
        return results
    
    # Test 5: Property segments
    results['property_segments'] = test_property_segments(df)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<25} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {total_tests - passed_tests} test(s) failed")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)

