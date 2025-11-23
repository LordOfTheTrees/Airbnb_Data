# Batch script to run all city-level analyses
# Run this from the Airbnb_Data directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RUNNING ALL CITY-LEVEL ANALYSES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Attributes (City Level Analysis)
Write-Host "[1/6] Running City Level Analysis (Attributes)..." -ForegroundColor Yellow
python city_level_analysis.py -all
Write-Host ""

# 2. Occupancy Analysis
Write-Host "[2/6] Running Occupancy Comparison Analysis..." -ForegroundColor Yellow
python analyze_occupancy_comparison.py -all
Write-Host ""

# 3. Market Professionalization
Write-Host "[3/6] Running Market Professionalization Analysis..." -ForegroundColor Yellow
python analyze_market_professionalization.py -all
Write-Host ""

# 4. Professionalization Correlations
Write-Host "[4/6] Running Professionalization Correlations..." -ForegroundColor Yellow
python analyze_professionalization_correlations.py -all
Write-Host ""

# 5. Market Entry Barriers
Write-Host "[5/6] Running Market Entry Barriers Analysis..." -ForegroundColor Yellow
python analyze_market_entry_barriers.py -all
Write-Host ""

# 6. ROI Visualizations
Write-Host "[6/6] Running ROI Visualizations..." -ForegroundColor Yellow
python visualize_roi_results.py -all
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "ALL ANALYSES COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

