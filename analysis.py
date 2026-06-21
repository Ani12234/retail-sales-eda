import pandas as pd
import numpy as np
import json
import os

def analyze_data():
    clean_path = "retail_sales_clean.csv"
    if not os.path.exists(clean_path):
        print(f"Error: Clean dataset not found at {clean_path}! Please run data_cleaning.py first.")
        return
        
    print(f"Loading cleaned dataset from {clean_path}...")
    df = pd.read_csv(clean_path)
    
    # 1. Group Revenue by Product Category and Region
    print("\n--- 1. Revenue by Product Category and Region ---")
    cat_region_rev = df.groupby(['product_category', 'region'])['revenue'].sum().unstack(fill_value=0)
    print(cat_region_rev.to_markdown(floatfmt=",.2f"))
    
    # Summary of revenue by product category
    print("\n--- Revenue by Product Category ---")
    cat_rev = df.groupby('product_category')['revenue'].agg(['sum', 'count']).rename(columns={'sum': 'revenue', 'count': 'orders'})
    cat_rev['percentage'] = (cat_rev['revenue'] / cat_rev['revenue'].sum()) * 100
    print(cat_rev.to_markdown(floatfmt=",.2f"))
    
    # Summary of revenue by region
    print("\n--- Revenue by Region ---")
    region_rev = df.groupby('region')['revenue'].agg(['sum', 'count']).rename(columns={'sum': 'revenue', 'count': 'orders'})
    region_rev = region_rev.sort_values(by='revenue', ascending=False)
    region_rev['percentage'] = (region_rev['revenue'] / region_rev['revenue'].sum()) * 100
    print(region_rev.to_markdown(floatfmt=",.2f"))
    
    # 2. Q4 Revenue vs. Baseline (Q1-Q3 Average)
    print("\n--- 2. Q4 Revenue Comparison against Baseline (Q1-Q3 Average) ---")
    q_rev = df.groupby('quarter')['revenue'].sum()
    q_baseline_avg = (q_rev['Q1'] + q_rev['Q2'] + q_rev['Q3']) / 3
    q4_increase = (q_rev['Q4'] - q_baseline_avg) / q_baseline_avg * 100
    
    print(f"Q1 Revenue: ${q_rev['Q1']:,.2f}")
    print(f"Q2 Revenue: ${q_rev['Q2']:,.2f}")
    print(f"Q3 Revenue: ${q_rev['Q3']:,.2f}")
    print(f"Baseline Average (Q1-Q3): ${q_baseline_avg:,.2f}")
    print(f"Q4 Revenue: ${q_rev['Q4']:,.2f}")
    print(f"Q4 Revenue Increase vs. Baseline: {q4_increase:.2f}% (Target: ~23.00%)")
    
    # 3. 3 Lowest-Performing Regions by Revenue
    print("\n--- 3. Underperforming Regions Analysis (3 Lowest by Revenue) ---")
    lowest_3 = region_rev.nsmallest(3, 'revenue')
    total_rev = df['revenue'].sum()
    bottom_3_total_rev = lowest_3['revenue'].sum()
    bottom_3_share = (bottom_3_total_rev / total_rev) * 100
    
    print("Lowest 3 Regions:")
    for reg, row in lowest_3.iterrows():
        print(f"  {reg}: ${row['revenue']:,.2f} ({row['percentage']:.2f}% of total)")
    print(f"Combined Bottom 3 Revenue: ${bottom_3_total_rev:,.2f}")
    print(f"Combined Bottom 3 Revenue Share (Loss): {bottom_3_share:.2f}% (Target: ~18.00%)")
    
    # Calculate Shortfall from Average Regional Performance
    avg_region_rev = total_rev / 10
    target_3_regions = avg_region_rev * 3
    shortfall = target_3_regions - bottom_3_total_rev
    shortfall_pct_total = (shortfall / total_rev) * 100
    print(f"Regional Target (Average, 10% each): ${avg_region_rev:,.2f} per region")
    print(f"Shortfall of Bottom 3 vs. Target: ${shortfall:,.2f} ({shortfall_pct_total:.2f}% of total revenue)")
    
    # 4. Save results to JSON
    summary_data = {
        "metrics": {
            "total_revenue": float(total_rev),
            "total_orders": int(len(df)),
            "q1_revenue": float(q_rev['Q1']),
            "q2_revenue": float(q_rev['Q2']),
            "q3_revenue": float(q_rev['Q3']),
            "q4_revenue": float(q_rev['Q4']),
            "q_baseline_average": float(q_baseline_avg),
            "q4_revenue_increase_pct": float(q4_increase),
            "bottom_3_regions": list(lowest_3.index),
            "bottom_3_revenue": float(bottom_3_total_rev),
            "bottom_3_revenue_share_pct": float(bottom_3_share),
            "shortfall_vs_avg_pct": float(shortfall_pct_total)
        },
        "revenue_by_category": cat_rev['revenue'].to_dict(),
        "revenue_by_region": region_rev['revenue'].to_dict()
    }
    
    summary_path = "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"\nSuccessfully exported analysis summary: {summary_path}")

if __name__ == "__main__":
    analyze_data()
