"""
Map Airbnb city names to Zillow metro areas.

This module provides a mapping dictionary to match Airbnb city folder names
to Zillow metro area names (RegionName).
"""


def get_city_to_zillow_mapping():
    """
    Create mapping dictionary from Airbnb city names to Zillow metro names.
    
    Returns:
        Dictionary mapping Airbnb city name -> Zillow RegionName (or None if unmatchable)
    """
    mapping = {
        # Direct matches
        'Austin': 'Austin, TX',
        'Chicago': 'Chicago, IL',
        'Dallas': 'Dallas, TX',
        'Denver': 'Denver, CO',
        'Seattle': 'Seattle, WA',
        'Nashville': 'Nashville, TN',
        'New_Orleans': 'New Orleans, LA',
        
        # Name transformations (underscore to space, add state)
        'New_York': 'New York, NY',
        'Los_Angeles': 'Los Angeles, CA',
        'San_Francisco': 'San Francisco, CA',
        'Washington_DC': 'Washington, DC',
        
        # Sub-cities that are part of larger metros
        'Jersey_City': 'New York, NY',  # Part of NYC metro
        'Cambridge': 'Boston, MA',  # Part of Boston metro
        'Oakland': 'San Francisco, CA',  # Part of SF metro
        
        # Special cases - need to check if these metros exist
        'Hawaii': 'Honolulu, HI',  # Assuming Hawaii refers to Honolulu
        'Oregon': 'Portland, OR',  # Assuming Oregon refers to Portland
        'Rhode_Island': 'Providence, RI',  # Assuming RI refers to Providence
        
        # Cities that might need manual checking
        'Albany': 'Albany, NY',
        'Asheville': 'Asheville, NC',
        'Bozeman': 'Bozeman, MT',
        'Columbus': 'Columbus, OH',  # Could be Columbus, GA but OH is more likely
        'Boston': 'Boston, MA',
        
        # Unmatchable (no Zillow data)
        'Paris': None,  # France - no Zillow data
    }
    
    return mapping


def match_city_to_zillow(city_name, zillow_data):
    """
    Match an Airbnb city name to a Zillow metro and return the metro data.
    
    Args:
        city_name: Airbnb city folder name (e.g., "New_York", "Austin")
        zillow_data: DataFrame from load_all_zillow_data()
        
    Returns:
        Series with Zillow metro data (RegionID, RegionName, zhvi_price, etc.)
        or None if no match found
    """
    mapping = get_city_to_zillow_mapping()
    
    # Get the Zillow metro name for this city
    zillow_metro_name = mapping.get(city_name)
    
    if zillow_metro_name is None:
        print(f"  WARNING: {city_name} has no Zillow mapping (e.g., Paris)")
        return None
    
    # Find the metro in Zillow data
    metro_data = zillow_data[zillow_data['RegionName'] == zillow_metro_name]
    
    if len(metro_data) == 0:
        print(f"  WARNING: {city_name} mapped to '{zillow_metro_name}' but not found in Zillow data")
        return None
    
    if len(metro_data) > 1:
        print(f"  WARNING: Multiple matches for '{zillow_metro_name}', using first")
    
    return metro_data.iloc[0]


def get_all_matched_cities(zillow_data):
    """
    Get mapping for all cities and show which ones have Zillow data.
    
    Args:
        zillow_data: DataFrame from load_all_zillow_data()
        
    Returns:
        Dictionary mapping city_name -> (matched, has_data)
    """
    mapping = get_city_to_zillow_mapping()
    results = {}
    
    for city_name, zillow_metro in mapping.items():
        if zillow_metro is None:
            results[city_name] = (False, False, None)
        else:
            metro_data = zillow_data[zillow_data['RegionName'] == zillow_metro]
            has_data = len(metro_data) > 0
            results[city_name] = (True, has_data, zillow_metro)
    
    return results


if __name__ == "__main__":
    # Test the mapping
    from load_zillow_data import load_all_zillow_data
    
    print("=" * 80)
    print("CITY TO ZILLOW MAPPING")
    print("=" * 80)
    
    zillow_data = load_all_zillow_data()
    results = get_all_matched_cities(zillow_data)
    
    print("\nMapping results:")
    print(f"{'City':<20} {'Zillow Metro':<30} {'Has Data':<10}")
    print("-" * 80)
    
    for city_name, (matched, has_data, metro_name) in sorted(results.items()):
        metro_display = metro_name if metro_name else "N/A (unmatchable)"
        status = "YES" if has_data else "NO"
        print(f"{city_name:<20} {metro_display:<30} {status:<10}")
    
    print("\n" + "=" * 80)
    print(f"Total cities: {len(results)}")
    print(f"Matched: {sum(1 for _, has_data, _ in results.values() if has_data)}")
    print(f"Unmatchable: {sum(1 for matched, _, _ in results.values() if not matched)}")
    print("=" * 80)

