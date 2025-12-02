"""
Cross-City Comparison Analysis Script
Compares markets across cities using aggregated data
Run AFTER city_level_analysis.py

This script loads data from all cities and creates comparative visualizations

⚠️ IMPORTANT: The detailed file is named listings.csv.gz (not listings_csv.gz)

Usage:
    python cross_city_analysis.py           # Uses simple 19-column listings.csv
    python cross_city_analysis.py -all      # Uses detailed 79-column listings.csv.gz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import io

# Import feature engineering functions
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from data.feature_engineering import apply_all_feature_engineering
from data.loaders import discover_city_folders

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

class CrossCityAnalyzer:
    """Analyze and compare Airbnb data across multiple cities"""
    
    def __init__(self, base_dir='.', use_detailed=False):
        self.base_dir = Path(base_dir)
        self.cities = {}
        self.combined_data = None
        self.city_summaries = {}
        self.use_detailed = use_detailed
        
    def load_all_cities(self, city_folders):
        """Load data from multiple city folders"""
        print("="*80)
        print("LOADING DATA FROM ALL CITIES")
        print("="*80)
        
        if self.use_detailed:
            print("🔍 MODE: DETAILED ANALYSIS (79 variables from listings_csv.gz)\n")
        else:
            print("🔍 MODE: SIMPLE ANALYSIS (19 variables from listings.csv)\n")
        
        for city in city_folders:
            city_path = self.base_dir / city
            
            # Choose file based on use_detailed flag
            if self.use_detailed:
                # When -all is specified, ONLY try the detailed .gz file
                listings_file = city_path / 'listings.csv.gz'
                file_type = "detailed (79 vars)"
                if not listings_file.exists():
                    listings_file = city_path / 'listings.csv'
                    file_type = "simple (19 vars) - fallback"
            else:
                # When -all is NOT specified, use the simple file
                listings_file = city_path / 'listings.csv'
                file_type = "simple (19 vars)"
                if not listings_file.exists():
                    listings_file = city_path / 'listings.csv.gz'
                    file_type = "detailed (79 vars) - fallback"
            
            if listings_file.exists():
                print(f"\n📍 Loading {city.upper()}...")
                try:
                    if str(listings_file).endswith('.gz'):
                        df = pd.read_csv(listings_file, compression='gzip')
                    else:
                        df = pd.read_csv(listings_file)
                    
                    df['city'] = city
                    self.cities[city] = df
                    print(f"   ✓ {len(df):,} listings × {len(df.columns)} columns [{file_type}]")
                    
                    # Load city-level summary if available
                    summary_file = Path('portfolio_outputs') / 'per_city' / city_name / 'analysis_output' / f'{city}_variable_summary.csv'
                    if summary_file.exists():
                        self.city_summaries[city] = pd.read_csv(summary_file)
                        print(f"   ✓ Loaded variable summary")
                    
                except Exception as e:
                    print(f"   ✗ Error: {e}")
            else:
                print(f"\n   ✗ No data found for {city}")
        
        if self.cities:
            self.combined_data = pd.concat(self.cities.values(), ignore_index=True)
            print(f"\n✅ Loaded {len(self.cities)} cities")
            print(f"   Total listings: {len(self.combined_data):,}")
            print(f"   Total columns: {len(self.combined_data.columns)}")
        
        return self
    
    def clean_data(self):
        """Clean and prepare combined data"""
        print("\n" + "="*80)
        print("CLEANING DATA")
        print("="*80)
        
        df = self.combined_data.copy()
        
        # Clean price
        if 'price' in df.columns:
            print("\n💵 Cleaning prices...")
            df['price_clean'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
            df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
            valid_prices = df['price_clean'].notna().sum()
            print(f"   ✓ {valid_prices:,} valid prices ({valid_prices/len(df)*100:.1f}%)")
        
        # Convert dates
        date_cols = ['last_review', 'first_review', 'host_since']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Apply feature engineering to combined data
        print("\n🔧 Applying feature engineering to combined data...")
        # Apply feature engineering (using "Combined" as city name for display)
        df = apply_all_feature_engineering(df, "Combined")
        
        self.combined_data = df
        print("\n✓ Data cleaning and feature engineering complete")
        return self
    
    def create_city_comparison_table(self):
        """Create comprehensive city comparison table"""
        print("\n" + "="*80)
        print("CREATING CITY COMPARISON TABLE")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Build aggregation dictionary
        agg_dict = {
            'id': 'count',
        }
        
        # Add available columns
        if 'price_clean' in df.columns:
            agg_dict['price_clean'] = ['mean', 'median', 'std', 'min', 'max']
        if 'number_of_reviews' in df.columns:
            agg_dict['number_of_reviews'] = ['mean', 'median', 'std']
        if 'reviews_per_month' in df.columns:
            agg_dict['reviews_per_month'] = ['mean', 'median']
        if 'availability_365' in df.columns:
            agg_dict['availability_365'] = ['mean', 'median']
        if 'minimum_nights' in df.columns:
            agg_dict['minimum_nights'] = ['mean', 'median']
        if 'calculated_host_listings_count' in df.columns:
            agg_dict['calculated_host_listings_count'] = ['mean', 'median']
        
        # Create summary
        city_comparison = df.groupby('city').agg(agg_dict).round(2)
        
        # Flatten column names
        city_comparison.columns = ['_'.join(col).strip() for col in city_comparison.columns.values]
        
        # Save
        city_comparison.to_csv('city_comparison_outputs/city_comparison_table.csv')
        print("✓ Saved: city_comparison_outputs/city_comparison_table.csv")
        
        # Print to console
        print("\n" + city_comparison.to_string())
        
        return city_comparison
    
    def create_comparison_visualizations(self):
        """Create visual comparisons across cities"""
        print("\n" + "="*80)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Figure 1: 4-Panel City Comparison
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Airbnb Market Comparison Across Cities', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Panel 1: Average Price
        ax = axes[0, 0]
        if 'price_clean' in df.columns:
            city_price = df.groupby('city')['price_clean'].mean().sort_values(ascending=False)
            city_price.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Average Nightly Price ($)', fontweight='bold', fontsize=11)
            ax.set_ylabel('')
            ax.set_title('Average Price by City', fontweight='bold', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
        
        # Panel 2: Market Size
        ax = axes[0, 1]
        city_size = df['city'].value_counts().sort_values(ascending=False)
        city_size.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Number of Listings', fontweight='bold', fontsize=11)
        ax.set_ylabel('')
        ax.set_title('Market Size (Total Listings)', fontweight='bold', fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        
        # Panel 3: Average Reviews (Popularity)
        ax = axes[1, 0]
        if 'number_of_reviews' in df.columns:
            city_reviews = df.groupby('city')['number_of_reviews'].mean().sort_values(ascending=False)
            city_reviews.plot(kind='barh', ax=ax, color='green')
            ax.set_xlabel('Average # of Reviews', fontweight='bold', fontsize=11)
            ax.set_ylabel('')
            ax.set_title('Average Reviews by City (Popularity Indicator)', 
                        fontweight='bold', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
        
        # Panel 4: Availability
        ax = axes[1, 1]
        if 'availability_365' in df.columns:
            city_avail = df.groupby('city')['availability_365'].mean().sort_values(ascending=False)
            city_avail.plot(kind='barh', ax=ax, color='purple')
            ax.set_xlabel('Average Availability (days/year)', fontweight='bold', fontsize=11)
            ax.set_ylabel('')
            ax.set_title('Average Availability by City', fontweight='bold', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/city_comparison_charts.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: city_comparison_outputs/city_comparison_charts.png")
        plt.close()
    
    def create_scatter_plots(self):
        """Create working scatter plot comparisons"""
        print("\n" + "="*80)
        print("CREATING SCATTER PLOT ANALYSIS")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Filter to valid prices for better visualization
        if 'price_clean' in df.columns:
            df_plot = df[df['price_clean'].notna()].copy()
            # Remove extreme outliers
            df_plot = df_plot[
                df_plot['price_clean'].between(
                    df_plot['price_clean'].quantile(0.01),
                    df_plot['price_clean'].quantile(0.99)
                )
            ]
        else:
            df_plot = df.copy()
        
        # Create 2x2 scatter plot figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Key Relationships in Airbnb Data Across Cities', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Get unique cities for color palette
        cities = df_plot['city'].unique()
        n_cities = len(cities)
        
        # Use a color palette that can handle many cities
        if n_cities <= 10:
            colors = sns.color_palette("tab10", n_cities)
        elif n_cities <= 20:
            colors = sns.color_palette("tab20", n_cities)
        else:
            colors = sns.color_palette("husl", n_cities)
        
        city_colors = dict(zip(cities, colors))
        
        # Plot 1: Price vs Reviews (limit to first 10 cities for clarity)
        ax = axes[0, 0]
        if 'price_clean' in df_plot.columns and 'number_of_reviews' in df_plot.columns:
            cities_to_plot = sorted(cities)[:10]  # First 10 cities alphabetically
            for city in cities_to_plot:
                city_data = df_plot[df_plot['city'] == city]
                # Limit reviews for better viz
                city_data_viz = city_data[city_data['number_of_reviews'] <= 200]
                ax.scatter(city_data_viz['number_of_reviews'], 
                          city_data_viz['price_clean'],
                          alpha=0.4, s=15, label=city, color=city_colors[city])
            ax.set_xlabel('Number of Reviews', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
            ax.set_title('Price vs. Popularity (Reviews)', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 200)
            print("  ✓ Plot 1: Price vs Reviews")
        
        # Plot 2: Availability vs Price
        ax = axes[0, 1]
        if 'price_clean' in df_plot.columns and 'availability_365' in df_plot.columns:
            cities_to_plot = sorted(cities)[:10]
            for city in cities_to_plot:
                city_data = df_plot[df_plot['city'] == city]
                ax.scatter(city_data['availability_365'], 
                          city_data['price_clean'],
                          alpha=0.4, s=15, label=city, color=city_colors[city])
            ax.set_xlabel('Availability (days/year)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
            ax.set_title('Availability vs. Price', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            print("  ✓ Plot 2: Availability vs Price")
        
        # Plot 3: Minimum Nights vs Price
        ax = axes[1, 0]
        if 'price_clean' in df_plot.columns and 'minimum_nights' in df_plot.columns:
            cities_to_plot = sorted(cities)[:10]
            for city in cities_to_plot:
                city_data = df_plot[df_plot['city'] == city]
                # Limit minimum nights for better viz
                city_data_viz = city_data[city_data['minimum_nights'] <= 30]
                ax.scatter(city_data_viz['minimum_nights'], 
                          city_data_viz['price_clean'],
                          alpha=0.4, s=15, label=city, color=city_colors[city])
            ax.set_xlabel('Minimum Nights Required', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
            ax.set_title('Minimum Stay vs. Price', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 30)
            print("  ✓ Plot 3: Minimum Nights vs Price")
        
        # Plot 4: Reviews per Month vs Price
        ax = axes[1, 1]
        if 'price_clean' in df_plot.columns and 'reviews_per_month' in df_plot.columns:
            cities_to_plot = sorted(cities)[:10]
            df_plot_rpm = df_plot[df_plot['reviews_per_month'].notna()]
            for city in cities_to_plot:
                city_data = df_plot_rpm[df_plot_rpm['city'] == city]
                # Limit reviews per month for better viz
                city_data_viz = city_data[city_data['reviews_per_month'] <= 10]
                ax.scatter(city_data_viz['reviews_per_month'], 
                          city_data_viz['price_clean'],
                          alpha=0.4, s=15, label=city, color=city_colors[city])
            ax.set_xlabel('Reviews per Month', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
            ax.set_title('Review Rate vs. Price', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 10)
            print("  ✓ Plot 4: Reviews per Month vs Price")
        
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/scatter_plots_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: city_comparison_outputs/scatter_plots_comparison.png")
        plt.close()
    
    def create_room_type_analysis(self):
        """Analyze room type distribution and performance"""
        print("\n" + "="*80)
        print("ROOM TYPE ANALYSIS")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        if 'room_type' not in df.columns:
            print("⚠️  Room type data not available")
            return self
        
        # Room type summary
        agg_dict = {
            'id': 'count'
        }
        if 'price_clean' in df.columns:
            agg_dict['price_clean'] = ['mean', 'median']
        if 'number_of_reviews' in df.columns:
            agg_dict['number_of_reviews'] = ['mean', 'median']
        
        room_summary = df.groupby('room_type').agg(agg_dict).round(2)
        room_summary.columns = ['_'.join(col).strip() for col in room_summary.columns.values]
        room_summary.to_csv('city_comparison_outputs/room_type_analysis.csv')
        print("✓ Saved: city_comparison_outputs/room_type_analysis.csv")
        
        # Room type distribution by city
        room_dist = pd.crosstab(df['city'], df['room_type'])
        room_dist.to_csv('city_comparison_outputs/room_type_by_city.csv')
        print("✓ Saved: city_comparison_outputs/room_type_by_city.csv")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Stacked bar chart of room types by city
        ax = axes[0]
        room_pct = room_dist.div(room_dist.sum(axis=1), axis=0) * 100
        room_pct.plot(kind='barh', stacked=True, ax=ax, 
                     colormap='Set3', width=0.8)
        ax.set_xlabel('Percentage (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Room Type Distribution by City', fontweight='bold', fontsize=14)
        ax.legend(title='Room Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Box plot of price by room type
        ax = axes[1]
        if 'price_clean' in df.columns:
            df_box = df[df['price_clean'].notna()].copy()
            df_box = df_box[
                df_box['price_clean'].between(
                    df_box['price_clean'].quantile(0.05),
                    df_box['price_clean'].quantile(0.95)
                )
            ]
            df_box.boxplot(column='price_clean', by='room_type', ax=ax)
            ax.set_xlabel('Room Type', fontweight='bold')
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Price Distribution by Room Type', fontweight='bold', fontsize=14)
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
            ax.get_figure().suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/room_type_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: city_comparison_outputs/room_type_visualizations.png")
        plt.close()
        
        return self
    
    def create_correlation_heatmap(self):
        """Create overall correlation heatmap"""
        print("\n" + "="*80)
        print("CREATING OVERALL CORRELATION MATRIX")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        key_vars = ['price_clean', 'number_of_reviews', 'availability_365',
                   'minimum_nights', 'reviews_per_month', 'calculated_host_listings_count',
                   'number_of_reviews_ltm']
        available_vars = [v for v in key_vars if v in numeric_cols]
        
        if len(available_vars) < 2:
            print("⚠️  Not enough numeric variables for correlation")
            return self
        
        # Calculate correlation
        corr_matrix = df[available_vars].corr()
        
        # Save CSV
        corr_matrix.to_csv('city_comparison_outputs/overall_correlation_matrix.csv')
        print("✓ Saved: city_comparison_outputs/overall_correlation_matrix.csv")

        # Create heatmap (no numbers on chart for readability)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - All Cities Combined', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/overall_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: city_comparison_outputs/overall_correlation_heatmap.png")
        plt.close()
        
        return self
    
    def create_size_vs_logprice_scatter(self):
        """Create scatter plots: size metrics vs log_price across all cities"""
        print("\n" + "="*80)
        print("CREATING SIZE vs LOG PRICE SCATTER PLOTS (ALL CITIES)")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Define variables
        size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
        price_var = 'log_price'
        
        # Check which variables are available
        available_size = [v for v in size_vars if v in df.columns]
        
        if price_var not in df.columns:
            print(f"WARNING: {price_var} not found in data")
            return self
        
        if not available_size:
            print("WARNING: No size variables found in data")
            return self
        
        print(f"\nSize variables: {', '.join(available_size)}")
        print(f"Price variable: {price_var}")
        
        # Filter out missing values
        plot_vars = available_size + [price_var, 'city']
        df_plot = df[plot_vars].dropna()
        
        # Filter to 95th percentile to focus on main data
        for var in available_size:
            p95 = df_plot[var].quantile(0.95)
            df_plot = df_plot[df_plot[var] <= p95]
        
        print(f"\nUsing {len(df_plot):,} listings with complete data")
        
        # Create figure with subplots: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get unique cities for color palette
        cities = df_plot['city'].unique()
        n_cities = len(cities)
        
        # Use a color palette that can handle many cities
        if n_cities <= 10:
            colors = sns.color_palette("tab10", n_cities)
        elif n_cities <= 20:
            colors = sns.color_palette("tab20", n_cities)
        else:
            colors = sns.color_palette("husl", n_cities)
        
        city_colors = dict(zip(cities, colors))
        
        # Create scatter plots
        for i, size_var in enumerate(available_size):
            if i >= 4:
                break
                
            ax = axes[i]
            
            # Plot each city with different color
            for city in sorted(cities):
                city_data = df_plot[df_plot['city'] == city]
                ax.scatter(city_data[size_var], city_data[price_var], 
                          alpha=0.3, s=15, label=city, color=city_colors[city])
            
            ax.set_xlabel(size_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=12)
            ax.set_ylabel('Log Price', fontweight='bold', fontsize=12)
            ax.set_title(f'{size_var.replace("_", " ").title()} vs Log Price', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend (only for first plot to avoid clutter)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        
        fig.suptitle('Property Size vs Log Price Across All Cities', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/size_vs_logprice_scatter.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: city_comparison_outputs/size_vs_logprice_scatter.png")
        plt.close()
        
        return self
    
    def create_occupancy_hexbin(self):
        """Create hexbin density plots: size metrics vs occupancy across all cities"""
        print("\n" + "="*80)
        print("CREATING SIZE vs OCCUPANCY HEXBIN PLOTS (ALL CITIES)")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Define variables
        size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
        # Prefer occupancy_rate (from estimated_occupancy_l365d), fallback to calendar_unavailable_proxy
        if 'occupancy_rate' in df.columns:
            occupancy_var = 'occupancy_rate'
        elif 'calendar_unavailable_proxy' in df.columns:
            occupancy_var = 'calendar_unavailable_proxy'
            print("WARNING: Using calendar_unavailable_proxy (less accurate than occupancy_rate)")
        else:
            print("WARNING: No occupancy metrics found in data")
            return self
        
        # Check which variables are available
        available_size = [v for v in size_vars if v in df.columns]
        
        if occupancy_var not in df.columns:
            print(f"WARNING: {occupancy_var} not found in data")
            return self
        
        if not available_size:
            print("WARNING: No size variables found in data")
            return self
        
        print(f"\nSize variables: {', '.join(available_size)}")
        
        # Filter out missing values
        plot_vars = available_size + [occupancy_var]
        df_plot = df[plot_vars].dropna()
        
        # Filter to 95th percentile
        for var in available_size:
            p95 = df_plot[var].quantile(0.95)
            df_plot = df_plot[df_plot[var] <= p95]
        
        print(f"\nUsing {len(df_plot):,} listings with complete data")
        
        # Create figure with subplots: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Create hexbin plots
        for i, size_var in enumerate(available_size):
            if i >= 4:
                break
                
            ax = axes[i]
            
            # Create hexbin plot
            hb = ax.hexbin(df_plot[size_var], df_plot[occupancy_var], 
                          gridsize=25, cmap='YlOrRd', mincnt=1)
            
            ax.set_xlabel(size_var.replace('_', ' ').title(), 
                         fontweight='bold', fontsize=12)
            ax.set_ylabel('Occupancy Rate (0-1)', 
                         fontweight='bold', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{size_var.replace("_", " ").title()} vs Occupancy (Density)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Count', fontweight='bold')
        
        fig.suptitle('Occupancy Density by Property Size (All Cities Combined)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/occupancy_hexbin.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: city_comparison_outputs/occupancy_hexbin.png")
        plt.close()
        
        return self
    
    def create_occupancy_boxplots(self):
        """Create box plots: occupancy by size categories across all cities"""
        print("\n" + "="*80)
        print("CREATING SIZE vs OCCUPANCY BOX PLOTS (ALL CITIES)")
        print("="*80)
        
        import os
        os.makedirs('portfolio_outputs/cross_city', exist_ok=True)
        
        df = self.combined_data
        
        # Define variables
        size_vars = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
        # Prefer occupancy_rate (from estimated_occupancy_l365d), fallback to calendar_unavailable_proxy
        if 'occupancy_rate' in df.columns:
            occupancy_var = 'occupancy_rate'
        elif 'calendar_unavailable_proxy' in df.columns:
            occupancy_var = 'calendar_unavailable_proxy'
            print("WARNING: Using calendar_unavailable_proxy (less accurate than occupancy_rate)")
        else:
            print("WARNING: No occupancy metrics found in data")
            return self
        
        # Check which variables are available
        available_size = [v for v in size_vars if v in df.columns]
        
        if occupancy_var not in df.columns:
            print(f"WARNING: {occupancy_var} not found in data")
            return self
        
        if not available_size:
            print("WARNING: No size variables found in data")
            return self
        
        print(f"\nSize variables: {', '.join(available_size)}")
        
        # Filter out missing values
        plot_vars = available_size + [occupancy_var]
        df_plot = df[plot_vars].dropna().copy()
        
        print(f"\nUsing {len(df_plot):,} listings with complete data")
        
        # Create figure with subplots: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define binning strategy
        binning_strategies = {
            'accommodates': {
                'bins': [0, 2, 4, 6, 8, 20],
                'labels': ['1-2', '3-4', '5-6', '7-8', '9+'],
            },
            'bathrooms': {
                'bins': [0, 1, 2, 3, 4, 20],
                'labels': ['1', '1.5-2', '2.5-3', '3.5-4', '4+'],
            },
            'bedrooms': {
                'bins': [-0.5, 0.5, 1.5, 2.5, 3.5, 20],
                'labels': ['Studio', '1BR', '2BR', '3BR', '4+'],
            },
            'beds': {
                'bins': [0, 2, 4, 6, 8, 20],
                'labels': ['1-2', '3-4', '5-6', '7-8', '9+'],
            }
        }
        
        # Create box plots
        for i, size_var in enumerate(available_size):
            if i >= 4:
                break
                
            ax = axes[i]
            
            bins = binning_strategies[size_var]['bins']
            labels = binning_strategies[size_var]['labels']
            
            # Create bins
            df_plot['binned'] = pd.cut(df_plot[size_var], bins=bins, labels=labels, include_lowest=True)
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            for label in labels:
                subset = df_plot[df_plot['binned'] == label][occupancy_var]
                if len(subset) > 0:
                    box_data.append(subset.values)
                    box_labels.append(label)
            
            if len(box_data) > 0:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{size_var.replace("_", " ").title()} vs Occupancy', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Size Category', fontweight='bold', fontsize=11)
                ax.set_ylabel('Occupancy Rate (0-1)', fontweight='bold', fontsize=11)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Calculate and print median occupancy by bin
                medians = df_plot.groupby('binned')[occupancy_var].median()
                print(f"\n{size_var} - Median Occupancy by Category (All Cities):")
                for cat, med in medians.items():
                    count = len(df_plot[df_plot['binned'] == cat])
                    print(f"  {cat}: {med:.3f} (n={count:,})")
        
        fig.suptitle('Occupancy Distribution by Size Category (All Cities Combined)', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('city_comparison_outputs/occupancy_boxplots.png', 
                   dpi=300, bbox_inches='tight')
        print("\n✓ Saved: city_comparison_outputs/occupancy_boxplots.png")
        plt.close()
        
        return self
    
    def run_full_analysis(self):
        """Run complete cross-city analysis"""
        self.create_city_comparison_table()
        self.create_comparison_visualizations()
        self.create_scatter_plots()
        self.create_room_type_analysis()
        self.create_correlation_heatmap()
        
        # New visualizations
        self.create_size_vs_logprice_scatter()
        self.create_occupancy_hexbin()
        self.create_occupancy_boxplots()
        
        print("\n" + "="*80)
        print("✅ CROSS-CITY ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files in city_comparison_outputs/:")
        print("  1. city_comparison_table.csv")
        print("  2. city_comparison_charts.png")
        print("  3. scatter_plots_comparison.png")
        print("  4. room_type_analysis.csv")
        print("  5. room_type_by_city.csv")
        print("  6. room_type_visualizations.png")
        print("  7. overall_correlation_matrix.csv")
        print("  8. overall_correlation_heatmap.png")
        print("  9. size_vs_logprice_scatter.png (NEW)")
        print("  10. occupancy_hexbin.png (NEW)")
        print("  11. occupancy_boxplots.png (NEW)")
        
        return self

# discover_city_folders is now imported from data.loaders


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this script from your main Airbnb_Data directory
    AFTER running city_level_analysis.py
    
    Usage:
        python cross_city_analysis.py           # Uses simple 19-column datasets
        python cross_city_analysis.py -all      # Uses detailed 79-column datasets
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    
    # Automatically discover city folders
    city_folders = discover_city_folders(base_dir='.')
    
    if not city_folders:
        print("❌ ERROR: No city folders found!")
        print("   Make sure you're running this from the Airbnb_Data directory")
        print("   and that city folders contain listings.csv or listings.csv.gz")
        sys.exit(1)
    
    print("\n" + "#"*80)
    print("CROSS-CITY COMPARISON ANALYSIS")
    print("#"*80)
    print(f"\n📍 Discovered {len(city_folders)} cities:")
    for i, city in enumerate(city_folders, 1):
        print(f"   {i:2d}. {city}")
    
    if use_detailed:
        print("\n🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print("\n🔍 MODE: SIMPLE ANALYSIS (19 variables)")
        print("💡 TIP: Run with -all flag for full 79-variable analysis")
    
    # Initialize analyzer
    analyzer = CrossCityAnalyzer(base_dir='.', use_detailed=use_detailed)
    
    # Load and process data
    (analyzer
     .load_all_cities(city_folders)
     .clean_data()
     .run_full_analysis())
    
    print("\n" + "#"*80)
    print("ALL DONE! Check the 'portfolio_outputs/cross_city' folder for results.")
    print("#"*80)
    
    if use_detailed:
        print("\n💡 You analyzed DETAILED datasets with 79 variables including:")
        print("   - property_type, bedrooms, bathrooms, accommodates")
        print("   - review_scores_rating and 6 other quality metrics")
        print("   - estimated_revenue_l365d and estimated_occupancy_l365d")
        print("   - And 50+ more detailed variables!")