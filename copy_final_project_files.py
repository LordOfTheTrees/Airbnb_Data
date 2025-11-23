"""
Copy all important analysis files to desktop folder for team sharing.

This script copies all per-city analysis files and cross-city comparison files
to a well-organized folder structure on the desktop for easy sharing via OneDrive.
"""

import shutil
from pathlib import Path
import sys

# Destination folder on desktop
DEST_BASE = Path(r"C:\Users\Andre\Desktop\Final Project Data and Summary")

# Source base directory (where the script is run from)
SOURCE_BASE = Path('.')

# Files to copy per city (organized by category)
PER_CITY_FILES = {
    'Core Investment Analysis': [
        '{city}_roi_visualizations_primary.png',
        '{city}_property_segments.csv',
    ],
    'Property Attribute Analysis': [
        '{city}_size_vs_log_price.png',
        '{city}_size_vs_occupancy_boxplots.png',
        '{city}_linear_regression_results.csv',
        # Individual plots folder will be handled separately
    ],
    'Market Structure Analysis': [
        '{city}_market_entry_barriers.png',
        '{city}_professionalization_correlations.png',
        '{city}_occupancy_comparison.png',
    ],
    'Data Exploration': [
        '{city}_correlation_matrix.csv',
        '{city}_correlation_heatmap_full.png',
        '{city}_top_correlations.csv',
        '{city}_variable_summary.csv',
    ]
}

# Cross-city comparison files
CROSS_CITY_FILES = [
    'city_comparison_data.csv',
    'city_regression_coefficients_comparison.png',
    'city_metrics_comparison.png',
    'market_professionalization_ranking.csv',
    'all_cities_census_exploration.png',
]

# Additional cross-city files that might exist
ADDITIONAL_CROSS_CITY_FILES = [
    'city_comparison_charts.png',
    'cap_mass_balance_analysis.png',
    'market_professionalization_relationships.png',
]


def discover_city_folders(base_dir='.'):
    """Discover all city folders in the repository."""
    base_path = Path(base_dir)
    city_folders = []
    
    exclude_dirs = {
        '__pycache__', 'Census', 'Kaggle', 'Zillow', 'old_scripts',
        'city_comparison_outputs', '.git'
    }
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in exclude_dirs:
            if (item / 'listings.csv').exists() or (item / 'listings.csv.gz').exists():
                city_folders.append(item.name)
    
    return sorted(city_folders)


def copy_file_safe(source, dest, description=""):
    """Copy a file safely, creating directories if needed."""
    if not source.exists():
        return False
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(source, dest)
        if description:
            print(f"  [OK] {description}")
        return True
    except Exception as e:
        print(f"  [ERROR] Error copying {source.name}: {e}")
        return False


def copy_per_city_files(city_name, source_base, dest_base):
    """Copy all important files for a single city."""
    city_source_dir = source_base / city_name / 'analysis_output'
    city_dest_dir = dest_base / 'Per-City Analysis' / city_name
    
    if not city_source_dir.exists():
        print(f"  [WARNING] No analysis_output folder found for {city_name}")
        return 0
    
    files_copied = 0
    
    # Copy files by category
    for category, file_patterns in PER_CITY_FILES.items():
        category_dir = city_dest_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern in file_patterns:
            filename = pattern.format(city=city_name)
            source_file = city_source_dir / filename
            dest_file = category_dir / filename
            
            if copy_file_safe(source_file, dest_file, f"{category}: {filename}"):
                files_copied += 1
    
    # Copy individual plots folder if it exists
    individual_plots_source = city_source_dir / f'{city_name}_individual_plots'
    if individual_plots_source.exists() and individual_plots_source.is_dir():
        individual_plots_dest = city_dest_dir / 'Property Attribute Analysis' / f'{city_name}_individual_plots'
        try:
            shutil.copytree(individual_plots_source, individual_plots_dest, dirs_exist_ok=True)
            print(f"  [OK] Property Attribute Analysis: {city_name}_individual_plots/ (folder)")
            files_copied += len(list(individual_plots_source.glob('*')))
        except Exception as e:
            print(f"  [ERROR] Error copying individual plots folder: {e}")
    
    return files_copied


def copy_cross_city_files(source_base, dest_base):
    """Copy all cross-city comparison files."""
    source_dir = source_base / 'city_comparison_outputs'
    dest_dir = dest_base / 'Cross-City Comparison'
    
    if not source_dir.exists():
        print(f"  [WARNING] No city_comparison_outputs folder found")
        return 0
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    
    # Copy main comparison files
    for filename in CROSS_CITY_FILES:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        
        if copy_file_safe(source_file, dest_file, f"Cross-City: {filename}"):
            files_copied += 1
    
    # Copy additional files if they exist
    for filename in ADDITIONAL_CROSS_CITY_FILES:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        
        if source_file.exists():
            if copy_file_safe(source_file, dest_file, f"Cross-City: {filename}"):
                files_copied += 1
    
    return files_copied


def copy_analysis_summary(source_base, dest_base):
    """Copy the ANALYSIS_SUMMARY.md document."""
    source_file = source_base / 'ANALYSIS_SUMMARY.md'
    dest_file = dest_base / 'ANALYSIS_SUMMARY.md'
    
    if source_file.exists():
        if copy_file_safe(source_file, dest_file, "ANALYSIS_SUMMARY.md"):
            return 1
    return 0


def main():
    """Main execution function."""
    print("="*80)
    print("COPYING FINAL PROJECT FILES TO DESKTOP")
    print("="*80)
    print(f"\nSource: {SOURCE_BASE.absolute()}")
    print(f"Destination: {DEST_BASE}")
    print()
    
    # Create destination directory
    DEST_BASE.mkdir(parents=True, exist_ok=True)
    
    # Discover cities
    print("Discovering city folders...")
    city_folders = discover_city_folders(SOURCE_BASE)
    print(f"Found {len(city_folders)} cities: {', '.join(city_folders)}")
    print()
    
    # Copy per-city files
    print("="*80)
    print("COPYING PER-CITY ANALYSIS FILES")
    print("="*80)
    total_city_files = 0
    
    for city_name in city_folders:
        print(f"\n{city_name}:")
        files_copied = copy_per_city_files(city_name, SOURCE_BASE, DEST_BASE)
        total_city_files += files_copied
        if files_copied == 0:
            print(f"  [WARNING] No files copied (may not have been analyzed yet)")
    
    print(f"\n[OK] Copied {total_city_files} per-city files")
    
    # Copy cross-city comparison files
    print("\n" + "="*80)
    print("COPYING CROSS-CITY COMPARISON FILES")
    print("="*80)
    cross_city_count = copy_cross_city_files(SOURCE_BASE, DEST_BASE)
    print(f"\n[OK] Copied {cross_city_count} cross-city comparison files")
    
    # Copy analysis summary
    print("\n" + "="*80)
    print("COPYING ANALYSIS SUMMARY DOCUMENT")
    print("="*80)
    summary_count = copy_analysis_summary(SOURCE_BASE, DEST_BASE)
    print(f"\n[OK] Copied {summary_count} summary document(s)")
    
    # Final summary
    print("\n" + "="*80)
    print("COPY COMPLETE!")
    print("="*80)
    print(f"\nTotal files copied: {total_city_files + cross_city_count + summary_count}")
    print(f"\nFiles organized in: {DEST_BASE}")
    print("\nFolder structure:")
    print("  - Per-City Analysis/")
    print("    - {City Name}/")
    print("      - Core Investment Analysis/")
    print("      - Property Attribute Analysis/")
    print("      - Market Structure Analysis/")
    print("      - Data Exploration/")
    print("  - Cross-City Comparison/")
    print("  - ANALYSIS_SUMMARY.md")
    print("\n[OK] Ready to upload to OneDrive!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Copy interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

