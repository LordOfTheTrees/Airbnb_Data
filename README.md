# Airbnb Data Analysis

Big Data and Customer Analytics class final project - Analysis of Airbnb listing data across multiple cities.

## Project Structure

```
airbnb_analysis/
├── analysis/              # Core analysis modules
│   ├── city_level.py     # City-level analysis
│   ├── cross_city.py     # Cross-city comparisons
│   └── ...               # Other analysis scripts
├── data/                 # Data loading and feature engineering
│   ├── loaders.py        # Data loading functions
│   └── feature_engineering.py  # Feature engineering functions
├── visualization/        # Visualization utilities
│   └── styles.py        # Style configuration
├── utils/               # Utility functions
├── scripts/             # Executable scripts
│   ├── run_all_analyses.py  # Master script to run all analyses
│   └── ...              # Other utility scripts
├── portfolio_outputs/   # Analysis outputs (gitignored)
│   ├── per_city/        # Per-city analysis results
│   └── cross_city/      # Cross-city comparison results
└── {City}/              # City data folders (data files gitignored)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Data files (CSV/CSV.GZ) are gitignored. Place your city data in city folders:
   - Each city folder should contain `listings.csv` or `listings.csv.gz`
   - Data files are excluded from version control

## Usage

### Run All Analyses

```bash
# Simple analysis (19 variables)
python scripts/run_all_analyses.py

# Detailed analysis (79 variables)
python scripts/run_all_analyses.py -all
```

### Run Individual Analyses

```bash
# City-level analysis
python analysis/city_level.py                    # All cities, simple
python analysis/city_level.py -all               # All cities, detailed
python analysis/city_level.py Austin             # Single city
python analysis/city_level.py Austin -all        # Single city, detailed

# Cross-city comparison
python analysis/cross_city.py                    # Simple
python analysis/cross_city.py -all               # Detailed

# Other analyses
python analysis/analyze_occupancy_comparison.py -all
python analysis/analyze_market_professionalization.py -all
python analysis/analyze_market_entry_barriers.py -all
python analysis/visualize_roi_results.py -all
```

## Outputs

All analysis outputs are saved to `portfolio_outputs/`:
- Per-city results: `portfolio_outputs/per_city/{city}/analysis_output/`
- Cross-city results: `portfolio_outputs/cross_city/`

Outputs are gitignored but kept locally for portfolio use.

## Data Files

Data files (CSV, CSV.GZ) are excluded from version control via `.gitignore`. 
The repository contains only code and analysis outputs.

## Notes

- Run scripts from the project root directory
- City folders should be in the root directory
- Analysis outputs are organized in `portfolio_outputs/` for easy portfolio access
