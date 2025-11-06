"""
Airbnb Data Exploratory Analysis
Analyzes listings data to answer key business questions about rental market attractiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization parameters
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class AirbnbAnalyzer:
    """Comprehensive Airbnb data analysis for multiple cities"""
    
    def __init__(self, base_dir='.'):
        """
        Initialize analyzer
        
        Args:
            base_dir: Base directory containing city folders
        """
        self.base_dir = Path(base_dir)
        self.cities = {}
        self.combined_data = None
        
    def load_city_data(self, city_folders):
        """
        Load data from multiple city folders
        
        Args:
            city_folders: List of city folder names (e.g., ['austin', 'boston', 'paris'])
        """
        print("="*80)
        print("LOADING AIRBNB DATA FROM MULTIPLE CITIES")
        print("="*80)
        
        for city in city_folders:
            city_path = self.base_dir / city
            listings_file = city_path / 'listings_csv.gz'
            
            # Try compressed file first, then regular CSV
            if not listings_file.exists():
                listings_file = city_path / 'listings.csv'
            
            if listings_file.exists():
                print(f"\n📍 Loading {city.upper()}...")
                try:
                    if str(listings_file).endswith('.gz'):
                        df = pd.read_csv(listings_file, compression='gzip')
                    else:
                        df = pd.read_csv(listings_file)
                    
                    # Add city identifier
                    df['city'] = city
                    self.cities[city] = df
                    print(f"   ✓ Loaded {len(df):,} listings with {len(df.columns)} variables")
                except Exception as e:
                    print(f"   ✗ Error loading {city}: {e}")
            else:
                print(f"\n   ✗ No listings file found for {city} at {listings_file}")
        
        if self.cities:
            print(f"\n✅ Successfully loaded {len(self.cities)} cities")
            # Combine all data
            self.combined_data = pd.concat(self.cities.values(), ignore_index=True)
            print(f"   Total listings across all cities: {len(self.combined_data):,}")
        else:
            print("\n❌ No data loaded!")
        
        return self
    
    def clean_data(self):
        """Clean and prepare data for analysis"""
        print("\n" + "="*80)
        print("DATA CLEANING")
        print("="*80)
        
        if self.combined_data is None:
            print("No data to clean!")
            return self
        
        df = self.combined_data.copy()
        
        # Clean price column
        if 'price' in df.columns:
            print("\n💵 Cleaning price column...")
            df['price_clean'] = df['price'].replace('[\$,]', '', regex=True)
            df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')
            print(f"   ✓ Converted {df['price_clean'].notna().sum():,} prices")
        
        # Clean response rates
        for col in ['host_response_rate', 'host_acceptance_rate']:
            if col in df.columns:
                df[f'{col}_clean'] = df[col].str.replace('%', '').astype(float) / 100
        
        # Parse host_since to datetime
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_tenure_days'] = (pd.Timestamp.now() - df['host_since']).dt.days
        
        # Parse review dates
        for col in ['first_review', 'last_review']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create derived variables
        print("\n🔨 Creating derived variables...")
        
        # Revenue per bedroom (efficiency metric)
        if 'estimated_revenue_l365d' in df.columns and 'bedrooms' in df.columns:
            df['revenue_per_bedroom'] = df['estimated_revenue_l365d'] / df['bedrooms'].replace(0, np.nan)
        
        # Revenue per guest capacity
        if 'estimated_revenue_l365d' in df.columns and 'accommodates' in df.columns:
            df['revenue_per_guest'] = df['estimated_revenue_l365d'] / df['accommodates'].replace(0, np.nan)
        
        # Review recency (% of reviews in last 12 months)
        if 'number_of_reviews_ltm' in df.columns and 'number_of_reviews' in df.columns:
            df['review_recency'] = df['number_of_reviews_ltm'] / df['number_of_reviews'].replace(0, np.nan)
            df['review_recency'] = df['review_recency'].clip(0, 1)
        
        # Occupancy rate (if available)
        if 'estimated_occupancy_l365d' in df.columns:
            df['occupancy_rate'] = df['estimated_occupancy_l365d']
        
        # Is superhost
        if 'host_is_superhost' in df.columns:
            df['is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        
        # Active listing (has reviews in last 12 months)
        if 'number_of_reviews_ltm' in df.columns:
            df['is_active'] = (df['number_of_reviews_ltm'] > 0).astype(int)
        
        print(f"   ✓ Created derived variables")
        
        self.combined_data = df
        return self
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        df = self.combined_data
        
        # Overall statistics
        print(f"\n📊 OVERALL DATASET:")
        print(f"   Total listings: {len(df):,}")
        print(f"   Cities: {df['city'].nunique()}")
        print(f"   Date range: {df['last_scraped'].min() if 'last_scraped' in df.columns else 'N/A'}")
        
        # Statistics by city
        print(f"\n📍 STATISTICS BY CITY:")
        city_stats = df.groupby('city').agg({
            'id': 'count',
            'price_clean': ['mean', 'median', 'std'],
            'estimated_revenue_l365d': ['mean', 'median'] if 'estimated_revenue_l365d' in df.columns else 'id',
            'occupancy_rate': ['mean', 'median'] if 'occupancy_rate' in df.columns else 'id',
            'number_of_reviews': ['mean', 'median'],
            'review_scores_rating': 'mean' if 'review_scores_rating' in df.columns else 'id'
        }).round(2)
        print(city_stats)
        
        # Key economic variables
        print(f"\n💰 KEY ECONOMIC VARIABLES (All Cities):")
        economic_vars = ['price_clean', 'estimated_revenue_l365d', 'occupancy_rate', 
                        'availability_365', 'minimum_nights']
        for var in economic_vars:
            if var in df.columns:
                print(f"\n{var}:")
                print(df[var].describe().round(2))
        
        # Property characteristics
        print(f"\n🏠 PROPERTY CHARACTERISTICS:")
        
        if 'room_type' in df.columns:
            print(f"\nRoom Type Distribution:")
            print(df.groupby('city')['room_type'].value_counts().unstack(fill_value=0))
        
        if 'property_type' in df.columns:
            print(f"\nTop 10 Property Types (All Cities):")
            print(df['property_type'].value_counts().head(10))
        
        if 'bedrooms' in df.columns:
            print(f"\nBedrooms Distribution:")
            print(df['bedrooms'].value_counts().sort_index().head(10))
        
        # Quality metrics
        print(f"\n⭐ QUALITY METRICS:")
        quality_vars = ['review_scores_rating', 'review_scores_cleanliness', 
                       'review_scores_location', 'review_scores_value']
        for var in quality_vars:
            if var in df.columns:
                print(f"\n{var}: Mean={df[var].mean():.2f}, Median={df[var].median():.2f}")
        
        # Host characteristics
        print(f"\n👤 HOST CHARACTERISTICS:")
        if 'is_superhost' in df.columns:
            print(f"Superhost %: {df['is_superhost'].mean()*100:.1f}%")
        if 'calculated_host_listings_count' in df.columns:
            print(f"Avg listings per host: {df['calculated_host_listings_count'].mean():.1f}")
            print(f"Multi-property hosts: {(df['calculated_host_listings_count'] > 1).mean()*100:.1f}%")
        
        return self
    
    def correlation_analysis(self):
        """Analyze correlations between key variables"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        df = self.combined_data
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Focus on key variables
        key_vars = ['price_clean', 'estimated_revenue_l365d', 'occupancy_rate',
                   'number_of_reviews', 'review_scores_rating', 'availability_365',
                   'accommodates', 'bedrooms', 'beds', 'minimum_nights',
                   'reviews_per_month', 'calculated_host_listings_count']
        
        # Filter to available variables
        available_vars = [v for v in key_vars if v in numeric_cols]
        
        if len(available_vars) < 2:
            print("Not enough numeric variables for correlation analysis")
            return self
        
        # Calculate correlations
        corr_matrix = df[available_vars].corr()
        
        # Print top correlations with estimated_revenue
        if 'estimated_revenue_l365d' in available_vars:
            print("\n💰 CORRELATIONS WITH ESTIMATED REVENUE:")
            revenue_corr = corr_matrix['estimated_revenue_l365d'].sort_values(ascending=False)
            print(revenue_corr)
        
        # Print top correlations with price
        if 'price_clean' in available_vars:
            print("\n💵 CORRELATIONS WITH PRICE:")
            price_corr = corr_matrix['price_clean'].sort_values(ascending=False)
            print(price_corr)
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix: Key Airbnb Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved correlation heatmap to outputs/")
        plt.close()
        
        return self
    
    def scatter_plots(self):
        """Generate key scatter plots for analysis"""
        print("\n" + "="*80)
        print("GENERATING SCATTER PLOTS")
        print("="*80)
        
        df = self.combined_data
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Airbnb Market Analysis: Key Relationships', fontsize=18, fontweight='bold')
        
        # Plot 1: Price vs Reviews
        if 'price_clean' in df.columns and 'number_of_reviews' in df.columns:
            ax = axes[0, 0]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['number_of_reviews'], city_data['price_clean'], 
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Number of Reviews', fontweight='bold')
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Price vs. Popularity (Reviews)')
            ax.legend()
            ax.set_xlim(0, df['number_of_reviews'].quantile(0.95))
            ax.set_ylim(0, df['price_clean'].quantile(0.95))
            print("   ✓ Plot 1: Price vs Reviews")
        
        # Plot 2: Revenue vs Occupancy
        if 'estimated_revenue_l365d' in df.columns and 'occupancy_rate' in df.columns:
            ax = axes[0, 1]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['occupancy_rate'], city_data['estimated_revenue_l365d'],
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Occupancy Rate', fontweight='bold')
            ax.set_ylabel('Estimated Annual Revenue ($)', fontweight='bold')
            ax.set_title('Revenue vs. Occupancy Rate')
            ax.legend()
            print("   ✓ Plot 2: Revenue vs Occupancy")
        
        # Plot 3: Price vs Review Score
        if 'price_clean' in df.columns and 'review_scores_rating' in df.columns:
            ax = axes[0, 2]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['review_scores_rating'], city_data['price_clean'],
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Review Score Rating', fontweight='bold')
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Price vs. Quality (Review Score)')
            ax.legend()
            ax.set_ylim(0, df['price_clean'].quantile(0.95))
            print("   ✓ Plot 3: Price vs Review Score")
        
        # Plot 4: Bedrooms vs Revenue
        if 'bedrooms' in df.columns and 'estimated_revenue_l365d' in df.columns:
            ax = axes[1, 0]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['bedrooms'], city_data['estimated_revenue_l365d'],
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Number of Bedrooms', fontweight='bold')
            ax.set_ylabel('Estimated Annual Revenue ($)', fontweight='bold')
            ax.set_title('Property Size vs. Revenue')
            ax.legend()
            ax.set_xlim(0, 6)
            print("   ✓ Plot 4: Bedrooms vs Revenue")
        
        # Plot 5: Availability vs Price
        if 'availability_365' in df.columns and 'price_clean' in df.columns:
            ax = axes[1, 1]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['availability_365'], city_data['price_clean'],
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Availability (365 days)', fontweight='bold')
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Availability vs. Price Strategy')
            ax.legend()
            ax.set_ylim(0, df['price_clean'].quantile(0.95))
            print("   ✓ Plot 5: Availability vs Price")
        
        # Plot 6: Accommodates vs Price
        if 'accommodates' in df.columns and 'price_clean' in df.columns:
            ax = axes[1, 2]
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                ax.scatter(city_data['accommodates'], city_data['price_clean'],
                          alpha=0.5, label=city, s=30)
            ax.set_xlabel('Accommodates (# of guests)', fontweight='bold')
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Guest Capacity vs. Price')
            ax.legend()
            ax.set_xlim(0, 12)
            ax.set_ylim(0, df['price_clean'].quantile(0.95))
            print("   ✓ Plot 6: Accommodates vs Price")
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/scatter_plots_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved scatter plots to outputs/")
        plt.close()
        
        return self
    
    def property_type_analysis(self):
        """Analyze performance by property type"""
        print("\n" + "="*80)
        print("PROPERTY TYPE ANALYSIS")
        print("="*80)
        
        df = self.combined_data
        
        if 'property_type' not in df.columns:
            print("Property type data not available")
            return self
        
        # Focus on top property types
        top_types = df['property_type'].value_counts().head(10).index
        df_filtered = df[df['property_type'].isin(top_types)]
        
        # Calculate metrics by property type
        if 'estimated_revenue_l365d' in df.columns:
            print("\n💰 REVENUE BY PROPERTY TYPE:")
            revenue_by_type = df_filtered.groupby('property_type')['estimated_revenue_l365d'].agg([
                'count', 'mean', 'median', 'std'
            ]).round(2).sort_values('mean', ascending=False)
            print(revenue_by_type)
            
            # Create box plot
            plt.figure(figsize=(14, 8))
            df_top10 = df_filtered[df_filtered['estimated_revenue_l365d'].notna()]
            df_top10 = df_top10[df_top10['estimated_revenue_l365d'] < df_top10['estimated_revenue_l365d'].quantile(0.95)]
            
            sns.boxplot(data=df_top10, y='property_type', x='estimated_revenue_l365d', 
                       order=revenue_by_type.index)
            plt.xlabel('Estimated Annual Revenue ($)', fontweight='bold')
            plt.ylabel('Property Type', fontweight='bold')
            plt.title('Revenue Distribution by Property Type (Top 10)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('/mnt/user-data/outputs/revenue_by_property_type.png', dpi=300, bbox_inches='tight')
            print("\n✅ Saved property type analysis to outputs/")
            plt.close()
        
        # Room type analysis
        if 'room_type' in df.columns:
            print("\n🏠 METRICS BY ROOM TYPE:")
            room_metrics = df.groupby('room_type').agg({
                'price_clean': ['count', 'mean', 'median'],
                'estimated_revenue_l365d': ['mean', 'median'] if 'estimated_revenue_l365d' in df.columns else 'price_clean',
                'review_scores_rating': 'mean' if 'review_scores_rating' in df.columns else 'price_clean'
            }).round(2)
            print(room_metrics)
        
        return self
    
    def market_comparison(self):
        """Compare markets across cities"""
        print("\n" + "="*80)
        print("MARKET COMPARISON ACROSS CITIES")
        print("="*80)
        
        df = self.combined_data
        
        # City-level metrics
        city_metrics = df.groupby('city').agg({
            'id': 'count',
            'price_clean': ['mean', 'median'],
            'estimated_revenue_l365d': ['mean', 'median'] if 'estimated_revenue_l365d' in df.columns else 'id',
            'occupancy_rate': 'mean' if 'occupancy_rate' in df.columns else 'id',
            'number_of_reviews': 'mean',
            'review_scores_rating': 'mean' if 'review_scores_rating' in df.columns else 'id',
            'availability_365': 'mean'
        }).round(2)
        
        print("\n📊 CITY COMPARISON METRICS:")
        print(city_metrics)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('City Market Comparison', fontsize=18, fontweight='bold')
        
        cities = df['city'].unique()
        
        # Average price by city
        ax = axes[0, 0]
        city_prices = df.groupby('city')['price_clean'].mean().sort_values(ascending=False)
        city_prices.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('Average Price ($)', fontweight='bold')
        ax.set_title('Average Nightly Price by City')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Average revenue by city
        if 'estimated_revenue_l365d' in df.columns:
            ax = axes[0, 1]
            city_revenue = df.groupby('city')['estimated_revenue_l365d'].mean().sort_values(ascending=False)
            city_revenue.plot(kind='bar', ax=ax, color='darkgreen')
            ax.set_ylabel('Average Annual Revenue ($)', fontweight='bold')
            ax.set_title('Average Revenue by City')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Number of listings by city
        ax = axes[1, 0]
        city_counts = df['city'].value_counts().sort_values(ascending=False)
        city_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_ylabel('Number of Listings', fontweight='bold')
        ax.set_title('Market Size (Number of Listings)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Occupancy rate by city
        if 'occupancy_rate' in df.columns:
            ax = axes[1, 1]
            city_occupancy = df.groupby('city')['occupancy_rate'].mean().sort_values(ascending=False)
            city_occupancy.plot(kind='bar', ax=ax, color='purple')
            ax.set_ylabel('Average Occupancy Rate', fontweight='bold')
            ax.set_title('Occupancy Rate by City')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/city_market_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved city comparison to outputs/")
        plt.close()
        
        return self
    
    def export_summary(self):
        """Export summary statistics to CSV"""
        print("\n" + "="*80)
        print("EXPORTING SUMMARY DATA")
        print("="*80)
        
        df = self.combined_data
        
        # City-level summary
        city_summary = df.groupby('city').agg({
            'id': 'count',
            'price_clean': ['mean', 'median', 'std'],
            'estimated_revenue_l365d': ['mean', 'median', 'std'] if 'estimated_revenue_l365d' in df.columns else 'id',
            'occupancy_rate': ['mean', 'median'] if 'occupancy_rate' in df.columns else 'id',
            'number_of_reviews': ['mean', 'median'],
            'review_scores_rating': ['mean', 'std'] if 'review_scores_rating' in df.columns else 'id',
            'availability_365': 'mean'
        }).round(2)
        
        city_summary.to_csv('/mnt/user-data/outputs/city_summary_statistics.csv')
        print("✅ Saved city_summary_statistics.csv")
        
        # Property type summary
        if 'property_type' in df.columns:
            property_summary = df.groupby('property_type').agg({
                'id': 'count',
                'price_clean': ['mean', 'median'],
                'estimated_revenue_l365d': ['mean', 'median'] if 'estimated_revenue_l365d' in df.columns else 'id'
            }).round(2).sort_values(('id', 'count'), ascending=False)
            
            property_summary.to_csv('/mnt/user-data/outputs/property_type_summary.csv')
            print("✅ Saved property_type_summary.csv")
        
        # Export cleaned dataset sample
        sample_columns = ['city', 'property_type', 'room_type', 'price_clean', 
                         'estimated_revenue_l365d', 'occupancy_rate', 
                         'bedrooms', 'accommodates', 'number_of_reviews', 
                         'review_scores_rating', 'availability_365']
        available_columns = [col for col in sample_columns if col in df.columns]
        
        df[available_columns].head(1000).to_csv('/mnt/user-data/outputs/data_sample.csv', index=False)
        print("✅ Saved data_sample.csv (first 1000 rows)")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files in /mnt/user-data/outputs/:")
        print("  1. correlation_heatmap.png")
        print("  2. scatter_plots_analysis.png")
        print("  3. revenue_by_property_type.png")
        print("  4. city_market_comparison.png")
        print("  5. city_summary_statistics.csv")
        print("  6. property_type_summary.csv")
        print("  7. data_sample.csv")
        
        return self


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    # If your data is organized as:
    # ./austin/listings_csv.gz
    # ./boston/listings_csv.gz
    # ./paris/listings.csv
    
    analyzer = AirbnbAnalyzer(base_dir='.')
    
    # Load data from city folders
    analyzer.load_city_data(['austin', 'boston', 'chicago', 'denver', 
                            'los-angeles', 'new-york-city', 'paris', 
                            'portland', 'san-francisco', 'seattle'])
    
    # Run full analysis pipeline
    (analyzer
     .clean_data()
     .descriptive_statistics()
     .correlation_analysis()
     .scatter_plots()
     .property_type_analysis()
     .market_comparison()
     .export_summary())
    """
    
    print("\n" + "="*80)
    print("AIRBNB EXPLORATORY ANALYSIS SCRIPT")
    print("="*80)
    print("\nThis script provides comprehensive analysis of Airbnb data.")
    print("\nTo use:")
    print("1. Ensure your city folders contain 'listings_csv.gz' or 'listings.csv'")
    print("2. Update the city_folders list below with your city names")
    print("3. Run the script")
    print("\n" + "="*80)
    
    # ====== CUSTOMIZE THESE SETTINGS ======
    
    # List your city folder names here
    city_folders = ['Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge', 'Chicago',
                   'Columbus', 'Dallas', 'Denver', 'Hawaii', 'Jersey_City', 'Los_Angeles', 
                   'Nashville', 'New_Orleans', 'New_York', 'Oakland', 'Oregon', 'Paris',
                    'Paris', 'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC']

    # Base directory (change if needed)
    base_directory = '.'
    
    # ======================================
    
    # Initialize and run analysis
    try:
        analyzer = AirbnbAnalyzer(base_dir=base_directory)
        
        (analyzer
         .load_city_data(city_folders)
         .clean_data()
         .descriptive_statistics()
         .correlation_analysis()
         .scatter_plots()
         .property_type_analysis()
         .market_comparison()
         .export_summary())
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()