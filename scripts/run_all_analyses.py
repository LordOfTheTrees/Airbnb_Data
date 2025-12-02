"""
Master script to run all analyses
Python replacement for run_all_analyses.ps1

Usage:
    python scripts/run_all_analyses.py           # Simple analysis (19 vars)
    python scripts/run_all_analyses.py -all     # Detailed analysis (79 vars)
"""
import sys
import subprocess
from pathlib import Path

def run_script(script_name, use_detailed=False):
    """Run a Python script with optional -all flag"""
    script_path = Path('analysis') / script_name
    if not script_path.exists():
        print(f"  [WARNING] Script not found: {script_path}")
        return False
    
    cmd = ['python', str(script_path)]
    if use_detailed:
        cmd.append('-all')
    
    print(f"\n  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    use_detailed = '-all' in sys.argv
    
    print("="*80)
    print("RUNNING ALL ANALYSES")
    print("="*80)
    if use_detailed:
        print("MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print("MODE: SIMPLE ANALYSIS (19 variables)")
    print()
    
    scripts = [
        ('city_level.py', 'City Level Analysis (Attributes)'),
        ('analyze_occupancy_comparison.py', 'Occupancy Comparison Analysis'),
        ('analyze_market_professionalization.py', 'Market Professionalization Analysis'),
        ('analyze_professionalization_correlations.py', 'Professionalization Correlations'),
        ('analyze_market_entry_barriers.py', 'Market Entry Barriers Analysis'),
        ('visualize_roi_results.py', 'ROI Visualizations'),
    ]
    
    results = []
    for script_name, description in scripts:
        print(f"\n[{scripts.index((script_name, description)) + 1}/{len(scripts)}] {description}...")
        success = run_script(script_name, use_detailed)
        results.append((description, success))
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    for description, success in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {description}")
    
    print(f"\n  Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"  Failed: {failed}/{len(results)}")
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nCheck portfolio_outputs/ for results:")
    print("  - portfolio_outputs/per_city/{city}/analysis_output/")
    print("  - portfolio_outputs/cross_city/")

if __name__ == "__main__":
    main()

