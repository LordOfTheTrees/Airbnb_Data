"""
Remove all data files from city folders (they're backed up and gitignored)
"""
from pathlib import Path

# Data file patterns to remove
data_patterns = [
    'listings.csv',
    'listings.csv.gz',
    'calendar.csv.gz',
    'reviews.csv',
    'reviews.csv.gz',
    'neighbourhoods.csv',
    'neighbourhoods.geojson'
]

# Discover city folders
exclude_dirs = {'__pycache__', 'Census', 'Kaggle', 'Zillow', 'old_scripts', 
                'city_comparison_outputs', '.git', 'portfolio_outputs',
                'analysis', 'data', 'visualization', 'utils', 'scripts'}

city_folders = []
for item in Path('.').iterdir():
    if item.is_dir() and item.name not in exclude_dirs:
        city_folders.append(item.name)

print(f"Found {len(city_folders)} city folders")
print("Removing data files...\n")

total_removed = 0
for city in sorted(city_folders):
    city_path = Path(city)
    removed_count = 0
    
    for pattern in data_patterns:
        file_path = city_path / pattern
        if file_path.exists():
            file_path.unlink()
            removed_count += 1
            total_removed += 1
    
    if removed_count > 0:
        print(f"  {city}: Removed {removed_count} data file(s)")

print(f"\nTotal files removed: {total_removed}")
print("Data files cleaned up!")

