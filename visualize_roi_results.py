"""
Visualize ROI analysis results.

Generates:
1. ROI distribution by room type (histogram/violin plot)
3. ROI vs size (bedrooms) - box plot
4. Segment comparison chart - horizontal bar chart ranking segments
7. ROI vs occupancy scatter plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io
from scipy.stats import linregress
from city_level_analysis import apply_all_feature_engineering
from analyze_property_segments import load_city_data, create_size_bins

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def create_roi_visualizations(df, city_name, output_dir=None, use_primary=True):
    """
    Create ROI visualization suite for a city.
    
    Args:
        df: DataFrame with feature engineering applied (including ROI metrics)
        city_name: Name of city
        output_dir: Directory to save output files
        use_primary: If True, use PRIMARY metrics (calendar-based occupancy).
                     If False, use SECONDARY metrics (booked days only).
    """
    if output_dir is None:
        # Combined visualization goes to analysis_output
        combined_output_dir = Path(city_name) / 'analysis_output'
        # Individual charts go to exploration_output subfolder
        individual_output_dir = Path(city_name) / 'exploration_output' / 'roi_individual_charts'
    else:
        combined_output_dir = Path(output_dir)
        individual_output_dir = Path(output_dir) / 'roi_individual_charts'
    
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    individual_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CREATING ROI VISUALIZATIONS: {city_name.upper()}")
    print(f"{'='*80}")
    
    # Exclude hotel rooms from visualization
    # Rationale: Hotel rooms represent commercial operations, not residential rental investments
    if 'room_type' in df.columns:
        n_hotel = (df['room_type'] == 'Hotel room').sum()
        if n_hotel > 0:
            df = df[df['room_type'] != 'Hotel room'].copy()
            print(f"  🏨 Excluded {n_hotel:,} hotel room listings from visualization")
    
    # Select which metrics to use
    if use_primary:
        roi_col = 'cash_on_cash_roi'
        occupancy_col = 'occupancy_rate'
        revenue_col = 'est_annual_revenue'
        metric_label = 'PRIMARY (calendar-based occupancy)'
    else:
        roi_col = 'cash_on_cash_roi_booked'
        occupancy_col = 'occupancy_rate_booked'
        revenue_col = 'est_annual_revenue_booked'
        metric_label = 'SECONDARY (booked days only)'
    
    print(f"  Using {metric_label} metrics")
    
    # Filter to listings with valid ROI data
    if roi_col not in df.columns or occupancy_col not in df.columns:
        print(f"  ⚠️  Missing required columns for {metric_label}")
        print(f"     Required: {roi_col}, {occupancy_col}")
        return None
    
    valid_roi = df[roi_col].notna() & df[occupancy_col].notna()
    df_roi = df[valid_roi].copy()
    
    if len(df_roi) == 0:
        print(f"  ⚠️  No valid ROI data for {city_name}")
        return None
    
    print(f"  Listings with ROI data: {len(df_roi):,} ({len(df_roi)/len(df)*100:.1f}%)")
    
    # Create size bins if not already present
    if 'size_bin' not in df_roi.columns:
        df_roi['size_bin'] = create_size_bins(df_roi['bedrooms'])
    
    # Calculate robust scale limits (used across multiple charts)
    roi_values = df_roi[roi_col].dropna()
    p5 = roi_values.quantile(0.05)
    p95 = roi_values.quantile(0.95)
    iqr = roi_values.quantile(0.75) - roi_values.quantile(0.25)
    y_min = max(p5 - 1.5 * iqr, roi_values.min())
    y_max = min(p95 + 1.5 * iqr, roi_values.max())
    
    # Store individual chart figures for saving
    individual_figures = []
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ============================================================================
    # CHART 1: ROI DISTRIBUTION BY ROOM TYPE (Violin Plot)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    room_types = df_roi['room_type'].unique()
    colors = sns.color_palette("husl", len(room_types))
    room_type_colors = dict(zip(room_types, colors))
    
    # Create violin plot with clipped data for visualization
    for i, room_type in enumerate(room_types):
        room_data = df_roi[df_roi['room_type'] == room_type][roi_col]
        if len(room_data) > 0:
            # Clip extreme outliers for visualization (but keep them in calculations)
            room_data_clipped = room_data.clip(lower=y_min, upper=y_max)
            
            # Violin plot
            # Lines shown: mean (dashed) and median (solid) - these are the horizontal lines in each violin
            parts = ax1.violinplot([room_data_clipped], positions=[i], widths=0.6, 
                                   showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(room_type_colors[room_type])
                pc.set_alpha(0.7)
    
    ax1.set_xticks(range(len(room_types)))
    ax1.set_xticklabels(room_types, rotation=45, ha='right')
    ax1.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax1.set_title('1. ROI Distribution by Room Type\n(Mean: dashed line, Median: solid line)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(y_min, y_max)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add median labels (using actual median, not clipped)
    for i, room_type in enumerate(room_types):
        room_data = df_roi[df_roi['room_type'] == room_type][roi_col]
        if len(room_data) > 0:
            median_val = room_data.median()
            # Position label above the plot area if median is near top
            if median_val > y_max * 0.9:
                ax1.text(i, y_max * 0.95, f'{median_val:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax1.text(i, median_val, f'{median_val:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Create individual figure for Chart 1
    fig1 = plt.figure(figsize=(10, 6))
    ax1_ind = fig1.add_subplot(111)
    for i, room_type in enumerate(room_types):
        room_data = df_roi[df_roi['room_type'] == room_type]['cash_on_cash_roi']
        if len(room_data) > 0:
            room_data_clipped = room_data.clip(lower=y_min, upper=y_max)
            parts = ax1_ind.violinplot([room_data_clipped], positions=[i], widths=0.6, 
                                       showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(room_type_colors[room_type])
                pc.set_alpha(0.7)
    ax1_ind.set_xticks(range(len(room_types)))
    ax1_ind.set_xticklabels(room_types, rotation=45, ha='right')
    ax1_ind.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax1_ind.set_title(f'{city_name}: ROI Distribution by Room Type\n(Mean: dashed line, Median: solid line)', 
                     fontsize=14, fontweight='bold', pad=20)
    ax1_ind.set_ylim(y_min, y_max)
    ax1_ind.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1_ind.grid(True, alpha=0.3)
    ax1_ind.legend()
    for i, room_type in enumerate(room_types):
        room_data = df_roi[df_roi['room_type'] == room_type]['cash_on_cash_roi']
        if len(room_data) > 0:
            median_val = room_data.median()
            if median_val > y_max * 0.9:
                ax1_ind.text(i, y_max * 0.95, f'{median_val:.1f}%', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax1_ind.text(i, median_val, f'{median_val:.1f}%', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
    fig1.tight_layout()
    individual_figures.append(('1_roi_by_room_type', fig1))
    
    # ============================================================================
    # CHART 2: ROI VS SIZE (BEDROOMS) - Box Plot
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Order size bins
    size_order = ['Studio', '1BR', '2BR', '3+BR']
    df_roi['size_bin_ordered'] = pd.Categorical(df_roi['size_bin'], 
                                                categories=size_order + ['Unknown'], 
                                                ordered=True)
    
    # Create box plot with same scale limits as room type plot
    box_data = [df_roi[df_roi['size_bin'] == size][roi_col].values 
                for size in size_order if size in df_roi['size_bin'].values]
    box_labels = [size for size in size_order if size in df_roi['size_bin'].values]
    
    # Clip data for visualization (but keep outliers for box plot calculation)
    box_data_clipped = []
    for data in box_data:
        clipped = np.clip(data, y_min, y_max)
        box_data_clipped.append(clipped)
    
    # Box plot elements:
    # - Box: 25th to 75th percentile (IQR)
    # - Line in box: Median (50th percentile)
    # - Whiskers: Extend to 1.5*IQR or min/max
    # - Mean: Dashed line (when showmeans=True, meanline=True)
    bp = ax2.boxplot(box_data_clipped, labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True, showfliers=False)  # Hide outliers since we're clipping
    
    # Color boxes
    colors_box = sns.color_palette("Set2", len(box_data))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Size (Bedrooms)', fontsize=12, fontweight='bold')
    ax2.set_title('2. ROI Distribution by Property Size\n(Box: IQR, Line: Median, Dashed: Mean)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(y_min, y_max)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add sample size labels
    for i, size in enumerate(box_labels):
        n = len(df_roi[df_roi['size_bin'] == size])
        ax2.text(i+1, y_min + 0.02 * (y_max - y_min),
                f'n={n:,}', ha='center', fontsize=9, style='italic')
    
    # Create individual figure for Chart 2
    fig2 = plt.figure(figsize=(10, 6))
    ax2_ind = fig2.add_subplot(111)
    bp_ind = ax2_ind.boxplot(box_data_clipped, labels=box_labels, patch_artist=True, 
                             showmeans=True, meanline=True, showfliers=False)
    for patch, color in zip(bp_ind['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2_ind.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax2_ind.set_xlabel('Size (Bedrooms)', fontsize=12, fontweight='bold')
    ax2_ind.set_title(f'{city_name}: ROI Distribution by Property Size\n(Box: IQR, Line: Median, Dashed: Mean)', 
                     fontsize=14, fontweight='bold', pad=20)
    ax2_ind.set_ylim(y_min, y_max)
    ax2_ind.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2_ind.grid(True, alpha=0.3, axis='y')
    ax2_ind.legend()
    for i, size in enumerate(box_labels):
        n = len(df_roi[df_roi['size_bin'] == size])
        ax2_ind.text(i+1, y_min + 0.02 * (y_max - y_min),
                    f'n={n:,}', ha='center', fontsize=9, style='italic')
    fig2.tight_layout()
    individual_figures.append(('2_roi_by_size', fig2))
    
    # ============================================================================
    # CHART 3: SEGMENT COMPARISON CHART - Top Segments by ROI
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, :])
    
    # Calculate segment-level stats
    df_roi['segment'] = df_roi['room_type'].astype(str) + ' × ' + df_roi['size_bin'].astype(str)
    
    segment_stats = []
    for segment in df_roi['segment'].unique():
        segment_data = df_roi[df_roi['segment'] == segment]
        if len(segment_data) >= 10:  # Only include segments with at least 10 listings
            stats = {
                'segment': segment,
                'median_roi': segment_data[roi_col].median(),
                'n_listings': len(segment_data),
                'median_cap_rate': segment_data['cap_rate' if use_primary else 'cap_rate_booked'].median() if ('cap_rate' if use_primary else 'cap_rate_booked') in segment_data.columns else np.nan,
            }
            segment_stats.append(stats)
    
    segments_df = pd.DataFrame(segment_stats)
    if len(segments_df) > 0:
        segments_df = segments_df.sort_values('median_roi', ascending=True)
        segments_df = segments_df.tail(10)  # Top 10 segments
        
        # Create horizontal bar chart
        y_pos = np.arange(len(segments_df))
        colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(segments_df)))
        
        bars = ax3.barh(y_pos, segments_df['median_roi'], color=colors_bar, alpha=0.7)
        
        # Set x-axis limits with padding to prevent text overlap
        x_min = segments_df['median_roi'].min()
        x_max = segments_df['median_roi'].max()
        x_range = x_max - x_min
        x_padding = max(x_range * 0.15, 5)  # 15% padding or at least 5 units
        
        ax3.set_xlim(x_min - x_padding, x_max + x_padding)
        
        # Add value labels (positioned to the right of bars)
        for i, (idx, row) in enumerate(segments_df.iterrows()):
            # Position label to the right of the bar
            label_x = row['median_roi'] + (x_padding * 0.1) if row['median_roi'] >= 0 else row['median_roi'] - (x_padding * 0.1)
            ax3.text(label_x, i, f"{row['median_roi']:.1f}%", 
                    va='center', fontweight='bold', fontsize=10)
            # Add sample size on the left side
            ax3.text(x_min - x_padding * 0.9, i, f"n={row['n_listings']:,}", 
                    va='center', ha='right', fontsize=9, style='italic', color='gray')
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(segments_df['segment'], fontsize=10)
        ax3.set_xlabel('Median Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
        ax3.set_title('3. Top 10 Segments by ROI (with at least 10 listings)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.legend()
        
        # Create individual figure for Chart 3
        fig3 = plt.figure(figsize=(12, 8))
        ax3_ind = fig3.add_subplot(111)
        bars_ind = ax3_ind.barh(y_pos, segments_df['median_roi'], color=colors_bar, alpha=0.7)
        ax3_ind.set_xlim(x_min - x_padding, x_max + x_padding)
        for i, (idx, row) in enumerate(segments_df.iterrows()):
            label_x = row['median_roi'] + (x_padding * 0.1) if row['median_roi'] >= 0 else row['median_roi'] - (x_padding * 0.1)
            ax3_ind.text(label_x, i, f"{row['median_roi']:.1f}%", 
                        va='center', fontweight='bold', fontsize=10)
            ax3_ind.text(x_min - x_padding * 0.9, i, f"n={row['n_listings']:,}", 
                        va='center', ha='right', fontsize=9, style='italic', color='gray')
        ax3_ind.set_yticks(y_pos)
        ax3_ind.set_yticklabels(segments_df['segment'], fontsize=10)
        ax3_ind.set_xlabel('Median Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
        ax3_ind.set_title(f'{city_name}: Top 10 Segments by ROI (with at least 10 listings)', 
                         fontsize=14, fontweight='bold', pad=20)
        ax3_ind.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax3_ind.grid(True, alpha=0.3, axis='x')
        ax3_ind.legend()
        fig3.tight_layout()
        individual_figures.append(('3_top_segments', fig3))
    
    # ============================================================================
    # CHART 4: ROI VS OCCUPANCY - Scatter Plot
    # ============================================================================
    ax4 = fig.add_subplot(gs[2, :])
    
    # Clip ROI values for scatter plot to focus on center mass
    df_roi_scatter = df_roi.copy()
    df_roi_scatter[f'{roi_col}_clipped'] = df_roi_scatter[roi_col].clip(lower=y_min, upper=y_max)
    
    # Create scatter plot colored by room type
    for room_type in room_types:
        room_data = df_roi_scatter[df_roi_scatter['room_type'] == room_type]
        if len(room_data) > 0:
            ax4.scatter(room_data[occupancy_col] * 100, room_data[f'{roi_col}_clipped'],
                       alpha=0.5, s=30, label=room_type, 
                       color=room_type_colors[room_type], edgecolors='white', linewidth=0.5)
    
    ax4.set_xlabel('Occupancy Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax4.set_title('4. ROI vs Occupancy Rate (colored by room type, focused on center mass)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylim(y_min, y_max)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', framealpha=0.9)
    
    # Add correlation coefficient (using original, non-clipped data)
    if len(df_roi) > 1:
        corr = df_roi[occupancy_col].corr(df_roi[roi_col])
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax4.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
    
    # Add trend line with equation (using original data for calculation, but plot within clipped range)
    if len(df_roi) > 1:
        x_data = df_roi[occupancy_col].values * 100
        y_data = df_roi[roi_col].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Generate regression line
        x_trend = np.linspace(df_roi[occupancy_col].min() * 100, 
                            df_roi[occupancy_col].max() * 100, 100)
        y_trend = slope * x_trend + intercept
        # Clip trend line to visible range
        y_trend_clipped = np.clip(y_trend, y_min, y_max)
        ax4.plot(x_trend, y_trend_clipped, "r--", alpha=0.8, linewidth=2, label='Linear Fit')
        
        # Format equation
        if intercept >= 0:
            eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
        else:
            eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
        
        # Add equation to chart
        ax4.text(0.05, 0.85, f'Linear Fit: {eq_text}\nR² = {r_value**2:.4f}', 
                transform=ax4.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                verticalalignment='top')
    
    # Create individual figure for Chart 4
    fig4 = plt.figure(figsize=(12, 8))
    ax4_ind = fig4.add_subplot(111)
    for room_type in room_types:
        room_data = df_roi_scatter[df_roi_scatter['room_type'] == room_type]
        if len(room_data) > 0:
            ax4_ind.scatter(room_data[occupancy_col] * 100, room_data[f'{roi_col}_clipped'],
                           alpha=0.5, s=30, label=room_type, 
                           color=room_type_colors[room_type], edgecolors='white', linewidth=0.5)
    ax4_ind.set_xlabel('Occupancy Rate (%)', fontsize=12, fontweight='bold')
    ax4_ind.set_ylabel('Cash-on-Cash ROI (%)', fontsize=12, fontweight='bold')
    ax4_ind.set_title(f'{city_name}: ROI vs Occupancy Rate ({metric_label})', 
                     fontsize=14, fontweight='bold', pad=20)
    ax4_ind.set_ylim(y_min, y_max)
    ax4_ind.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4_ind.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4_ind.grid(True, alpha=0.3)
    ax4_ind.legend(loc='best', framealpha=0.9)
    if len(df_roi) > 1:
        corr = df_roi[occupancy_col].corr(df_roi[roi_col])
        ax4_ind.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax4_ind.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='top')
        x_data = df_roi[occupancy_col].values * 100
        y_data = df_roi[roi_col].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Generate regression line
        x_trend = np.linspace(df_roi[occupancy_col].min() * 100, 
                            df_roi[occupancy_col].max() * 100, 100)
        y_trend = slope * x_trend + intercept
        y_trend_clipped = np.clip(y_trend, y_min, y_max)
        ax4_ind.plot(x_trend, y_trend_clipped, "r--", alpha=0.8, linewidth=2, label='Linear Fit')
        
        # Format equation
        if intercept >= 0:
            eq_text = f'y = {slope:.4f}x + {intercept:.4f}'
        else:
            eq_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'
        
        # Add equation to chart
        ax4_ind.text(0.05, 0.85, f'Linear Fit: {eq_text}\nR² = {r_value**2:.4f}', 
                    transform=ax4_ind.transAxes, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    verticalalignment='top')
    fig4.tight_layout()
    individual_figures.append(('4_roi_vs_occupancy', fig4))
    
    # Overall title
    metric_suffix = '_primary' if use_primary else '_secondary'
    fig.suptitle(f'ROI Analysis: {city_name} ({metric_label})', fontsize=16, fontweight='bold', y=0.995)
    
    # Save combined figure to analysis_output (use fig.savefig, not plt.savefig)
    output_file = combined_output_dir / f'{city_name}_roi_visualizations{metric_suffix}.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved combined visualizations to {output_file}")
    plt.close(fig)
    
    # Save individual chart images for PowerPoint to exploration_output subfolder
    individual_files = []
    for chart_name, chart_fig in individual_figures:
        individual_file = individual_output_dir / f'{city_name}_{chart_name}{metric_suffix}.png'
        chart_fig.savefig(individual_file, dpi=150, bbox_inches='tight')
        individual_files.append(individual_file)
        plt.close(chart_fig)
        print(f"  ✓ Saved individual chart: {individual_file}")
    
    return output_file, individual_files


def visualize_all_cities(city_folders, base_dir='.', use_detailed=False, use_primary=True):
    """
    Create ROI visualizations for all cities.
    
    Args:
        city_folders: List of city folder names
        base_dir: Base directory
        use_detailed: If True, use detailed 79-variable datasets
        use_primary: If True, use PRIMARY metrics (calendar-based). If False, use SECONDARY (booked days).
    """
    metric_label = "PRIMARY (calendar-based)" if use_primary else "SECONDARY (booked days)"
    print(f"\n{'='*80}")
    print(f"ROI VISUALIZATION - ALL CITIES ({metric_label})")
    print(f"{'='*80}")
    
    results = []
    
    for city_name in city_folders:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {city_name}")
            print(f"{'='*80}")
            
            # Load and process city data
            df = load_city_data(city_name, base_dir, use_detailed)
            
            # Apply feature engineering (including Zillow and ROI)
            df = apply_all_feature_engineering(df, city_name, include_zillow=True)
            
            # Create visualizations
            result = create_roi_visualizations(df, city_name, use_primary=use_primary)
            
            if result:
                if isinstance(result, tuple):
                    output_file, individual_files = result
                    results.append((city_name, output_file, individual_files))
                else:
                    results.append((city_name, result, []))
            
        except Exception as e:
            print(f"  ❌ Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Successfully created visualizations for {len(results)} cities:")
    for result_item in results:
        if len(result_item) == 3:
            city_name, output_file, individual_files = result_item
            print(f"    - {city_name}:")
            print(f"        Combined: {output_file}")
            print(f"        Individual charts: {len(individual_files)} files")
        else:
            city_name, output_file = result_item
            print(f"    - {city_name}: {output_file}")
    
    return results


if __name__ == "__main__":
    """
    Run ROI visualizations.
    
    Usage:
        python visualize_roi_results.py                    # All cities, simple datasets
        python visualize_roi_results.py -all               # All cities, detailed datasets
        python visualize_roi_results.py Austin               # Just Austin, simple dataset
        python visualize_roi_results.py Austin -all         # Just Austin, detailed dataset
    """
    
    # Parse command-line arguments
    use_detailed = '-all' in sys.argv
    use_secondary = '-secondary' in sys.argv or '-booked' in sys.argv
    use_primary = not use_secondary  # Default to primary
    
    # Check if a specific city was requested
    city_args = [arg for arg in sys.argv[1:] if arg not in ['-all', '-secondary', '-booked']]
    single_city = city_args[0] if city_args else None
    
    # City list
    all_cities = [
        'Albany', 'Asheville', 'Austin', 'Bozeman', 'Cambridge',
        'Chicago', 'Columbus', 'Dallas', 'Denver', 'Hawaii',
        'Jersey_City', 'Los_Angeles', 'Nashville', 'New_Orleans',
        'New_York', 'Oakland', 'Oregon', 'Paris',
        'Rhode_Island', 'San_Francisco', 'Seattle', 'Washington_DC'
    ]
    
    # Determine which cities to analyze
    if single_city:
        city_folders = [c for c in all_cities if c.lower() == single_city.lower()]
        
        if not city_folders:
            print(f"\n❌ ERROR: City '{single_city}' not found in city list!")
            print(f"\nAvailable cities:")
            for city in all_cities:
                print(f"  - {city}")
            sys.exit(1)
        
        print(f"\n🎯 SINGLE CITY MODE: Visualizing {city_folders[0]} only")
    else:
        city_folders = all_cities
        print(f"\n📊 BATCH MODE: Visualizing all {len(city_folders)} cities")
    
    if use_detailed:
        print(f"🔍 MODE: DETAILED ANALYSIS (79 variables)")
    else:
        print(f"🔍 MODE: SIMPLE ANALYSIS (19 variables)")
    
    # Run visualizations (generate both primary and secondary if not specified)
    if '-both' in sys.argv:
        # Generate both versions
        print(f"\n{'='*80}")
        print(f"GENERATING BOTH PRIMARY AND SECONDARY VISUALIZATIONS")
        print(f"{'='*80}")
        results_primary = visualize_all_cities(city_folders, base_dir='.', use_detailed=use_detailed, use_primary=True)
        results_secondary = visualize_all_cities(city_folders, base_dir='.', use_detailed=use_detailed, use_primary=False)
        results = results_primary + results_secondary
    else:
        # Generate specified version
        results = visualize_all_cities(city_folders, base_dir='.', use_detailed=use_detailed, use_primary=use_primary)
    
    print(f"\n{'='*80}")
    print(f"ALL DONE! Check each city's 'analysis_output' folder for visualization files.")
    print(f"{'='*80}")

