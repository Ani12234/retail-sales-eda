import pandas as pd
import numpy as np
import datetime
import os

def generate_dataset():
    # Set seed for reproducibility
    np.random.seed(42)
    
    total_rows = 10000
    clean_rows = 9700
    outliers_count = 180
    duplicates_count = 60
    negatives_count = 60
    
    print("Generating base clean dataset (9,700 records)...")
    
    # 10 Regions and their target revenue/row shares
    regions = ['West', 'Southwest', 'Mid-Atlantic', 'Northeast', 'Southeast', 'Midwest', 'South', 'Northwest', 'Mountain', 'New England']
    region_shares = {
        'West': 0.15,
        'Southwest': 0.14,
        'Mid-Atlantic': 0.13,
        'Northeast': 0.12,
        'Southeast': 0.10,
        'Midwest': 0.09,
        'South': 0.09,
        'Northwest': 0.07,
        'Mountain': 0.06,
        'New England': 0.05
    }
    
    # Assign regions to 9700 rows based on exact row shares
    region_assignments = []
    for r in regions:
        count = int(clean_rows * region_shares[r])
        region_assignments.extend([r] * count)
    # If there's any rounding difference, fill it up (should be exactly 9700)
    while len(region_assignments) < clean_rows:
        region_assignments.append(regions[0])
    
    # Shuffle region assignments with seed
    region_assignments = np.array(region_assignments)
    np.random.shuffle(region_assignments)
    
    # 4 Quarters and their target shares for a 23.0% Q4 increase over Q1-Q3 average
    # Q1 = Q2 = Q3 = 1.0 / (3 + 1.23)
    # Q4 = 1.23 / (3 + 1.23)
    q_baseline = 1.0 / 4.23
    q_q4 = 1.23 / 4.23
    quarter_shares = {
        'Q1': q_baseline,
        'Q2': q_baseline,
        'Q3': q_baseline,
        'Q4': q_q4
    }
    
    # Assign quarters randomly for each region to maintain independent distributions
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    quarter_assignments = np.random.choice(
        quarters, 
        size=clean_rows, 
        p=[quarter_shares[q] for q in quarters]
    )
    
    # Generate dates corresponding to the quarters
    dates = []
    for q in quarter_assignments:
        if q == 'Q1':
            start_date = datetime.date(2025, 1, 1)
            days = 90
        elif q == 'Q2':
            start_date = datetime.date(2025, 4, 1)
            days = 91
        elif q == 'Q3':
            start_date = datetime.date(2025, 7, 1)
            days = 92
        else:
            start_date = datetime.date(2025, 10, 1)
            days = 92
        random_days = np.random.randint(0, days)
        dates.append(start_date + datetime.timedelta(days=int(random_days)))
    
    # Product categories (Electronics is the mode with 25%)
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports', 'Books']
    cat_probs = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    category_assignments = np.random.choice(categories, size=clean_rows, p=cat_probs)
    
    # Customer segments
    segments = ['Consumer', 'Corporate', 'Home Office']
    segment_probs = [0.50, 0.30, 0.20]
    segment_assignments = np.random.choice(segments, size=clean_rows, p=segment_probs)
    
    # Units sold (ensure 10% are exactly 1, others are 2-20)
    units_sold = np.random.randint(2, 21, size=clean_rows)
    is_one = np.random.rand(clean_rows) < 0.10
    units_sold[is_one] = 1
    
    # Base prices by category
    base_prices = {
        'Electronics': 450.0,
        'Clothing': 45.0,
        'Home & Kitchen': 120.0,
        'Beauty': 35.0,
        'Sports': 85.0,
        'Books': 22.0
    }
    
    # Create initial DataFrame
    df_clean = pd.DataFrame({
        'order_id': [f"ORD-{10001+i}" for i in range(clean_rows)],
        'order_date': dates,
        'product_category': category_assignments,
        'region': region_assignments,
        'quarter': quarter_assignments,
        'units_sold': units_sold,
        'customer_segment': segment_assignments
    })
    
    # Generate initial unit prices
    initial_prices = []
    for cat in df_clean['product_category']:
        # Base price with +/- 15% random variation
        base = base_prices[cat]
        variation = np.random.uniform(-0.15, 0.15) * base
        initial_prices.append(base + variation)
    df_clean['unit_price'] = np.round(initial_prices, 2)
    df_clean['revenue'] = df_clean['units_sold'] * df_clean['unit_price']
    
    print("Calibrating revenues using cell matrix-scaling...")
    # Target total revenue
    target_total_revenue = 4800000.0  # $4.8M
    
    # For each region and quarter cell, scale revenues to meet exact shares
    df_clean['unit_price'] = df_clean['unit_price'].astype(float)
    
    for r in regions:
        for q in quarters:
            cell_mask = (df_clean['region'] == r) & (df_clean['quarter'] == q)
            cell_indices = df_clean[cell_mask].index
            
            if len(cell_indices) == 0:
                continue
            
            # Target cell revenue
            target_cell_rev = target_total_revenue * quarter_shares[q] * region_shares[r]
            
            # Initial cell revenue
            cell_units = df_clean.loc[cell_indices, 'units_sold'].values
            cell_prices = df_clean.loc[cell_indices, 'unit_price'].values
            initial_cell_rev = np.sum(cell_units * cell_prices)
            
            # Scale factor
            sf = target_cell_rev / initial_cell_rev
            
            # Apply scaling
            scaled_prices = np.round(cell_prices * sf, 2)
            df_clean.loc[cell_indices, 'unit_price'] = scaled_prices
            
            # Greedy adjustments to resolve rounding discrepancies to the cent
            target_cents = int(round(target_cell_rev * 100))
            current_cents = int(round(np.sum(cell_units * scaled_prices) * 100))
            diff_cents = target_cents - current_cents
            
            if diff_cents != 0:
                step = 1 if diff_cents > 0 else -1
                remaining = abs(diff_cents)
                
                # Sort cell indices by units sold ascending to use units=1 rows first
                sorted_cell_indices = cell_indices[np.argsort(cell_units)]
                
                # Iterate and adjust
                for idx in sorted_cell_indices:
                    u = df_clean.loc[idx, 'units_sold']
                    if u <= remaining:
                        df_clean.loc[idx, 'unit_price'] += step * 0.01
                        remaining -= u
                    if remaining == 0:
                        break
                
                # If still remaining (unlikely since we have rows with units=1), adjust the smallest units row
                if remaining > 0:
                    idx = sorted_cell_indices[0]
                    df_clean.loc[idx, 'unit_price'] += step * (remaining / 100.0) / df_clean.loc[idx, 'units_sold']
                    df_clean.loc[idx, 'unit_price'] = round(df_clean.loc[idx, 'unit_price'], 4) # Allow extra precision if absolutely needed, though rare
    
    # Recompute revenue
    df_clean['revenue'] = np.round(df_clean['units_sold'] * df_clean['unit_price'], 2)
    
    # Verify calibration results
    total_rev = df_clean['revenue'].sum()
    print(f"Calibrated Total Revenue: ${total_rev:,.2f}")
    
    q_revs = df_clean.groupby('quarter')['revenue'].sum()
    q_baseline_avg = (q_revs['Q1'] + q_revs['Q2'] + q_revs['Q3']) / 3
    q4_increase = (q_revs['Q4'] - q_baseline_avg) / q_baseline_avg * 100
    print(f"Calibrated Q4 vs. Baseline: {q4_increase:.2f}% increase (Target: 23.00%)")
    
    r_revs = df_clean.groupby('region')['revenue'].sum()
    bottom_3 = r_revs.nsmallest(3)
    bottom_3_share = bottom_3.sum() / total_rev * 100
    print(f"Calibrated Bottom 3 Regions Share: {bottom_3_share:.2f}% (Target: 18.00%)")
    print(f"Bottom 3 Regions: {list(bottom_3.index)}")
    
    # -------------------------------------------------------------
    # Inject Missing Values (exactly 1,200 values scattered)
    # -------------------------------------------------------------
    print("Injecting 1,200 missing values scattered across columns...")
    
    # Identify modes and medians of the clean dataset
    mode_category = df_clean['product_category'].mode()[0]  # 'Electronics'
    mode_region = df_clean['region'].mode()[0]              # 'West'
    median_units = float(df_clean['units_sold'].median())
    median_revenue = float(df_clean['revenue'].median())
    
    # We want to inject 300 missing values per column.
    # To keep total/quarter/regional stats perfectly accurate after imputation:
    # 1. missing region: select 300 rows that have region = mode_region, set region to NaN. Imputation will restore to mode_region.
    # 2. missing product_category: select 300 rows with category = mode_category, set category to NaN. Imputation will restore to mode_category.
    # 3. missing units_sold: select 300 rows with units_sold = median_units, set to NaN. Imputation will restore to median_units.
    # 4. missing revenue: select 300 rows with units_sold = 1 and set their unit_price and revenue to median_revenue.
    #    Then set revenue to NaN. Imputation will restore them to median_revenue.
    
    # Let's select rows for each injection ensuring no overlap (to keep it clean)
    # Initialize mask of selected rows
    selected_indices = set()
    
    # 1. Missing Region
    region_indices = df_clean[(df_clean['region'] == mode_region) & (~df_clean.index.isin(selected_indices))].index
    chosen_region_idx = np.random.choice(region_indices, size=300, replace=False)
    selected_indices.update(chosen_region_idx)
    
    # 2. Missing Product Category
    cat_indices = df_clean[(df_clean['product_category'] == mode_category) & (~df_clean.index.isin(selected_indices))].index
    chosen_cat_idx = np.random.choice(cat_indices, size=300, replace=False)
    selected_indices.update(chosen_cat_idx)
    
    # 3. Missing Units Sold
    units_indices = df_clean[(df_clean['units_sold'] == median_units) & (~df_clean.index.isin(selected_indices))].index
    chosen_units_idx = np.random.choice(units_indices, size=300, replace=False)
    selected_indices.update(chosen_units_idx)
    
    # 4. Missing Revenue
    # Select 300 rows that have units_sold = 1
    # We must force their original values in df_clean to have revenue equal to median_revenue.
    u1_indices = df_clean[(df_clean['units_sold'] == 1) & (~df_clean.index.isin(selected_indices))].index
    chosen_rev_idx = np.random.choice(u1_indices, size=300, replace=False)
    selected_indices.update(chosen_rev_idx)
    
    for idx in chosen_rev_idx:
        old_rev = df_clean.loc[idx, 'revenue']
        diff = median_revenue - old_rev
        
        # Apply to df_clean
        df_clean.loc[idx, 'unit_price'] = median_revenue
        df_clean.loc[idx, 'revenue'] = median_revenue
        
        # Distribute this difference to other rows in the same cell (quarter, region) to keep cell total constant!
        r = df_clean.loc[idx, 'region']
        q = df_clean.loc[idx, 'quarter']
        cell_mask = (df_clean['region'] == r) & (df_clean['quarter'] == q) & (~df_clean.index.isin(chosen_rev_idx))
        other_indices = df_clean[cell_mask].index
        
        if len(other_indices) > 0:
            # subtract diff / total_units from unit_prices
            total_other_units = df_clean.loc[other_indices, 'units_sold'].sum()
            adjustment_per_unit = diff / total_other_units
            df_clean.loc[other_indices, 'unit_price'] -= adjustment_per_unit
            df_clean.loc[other_indices, 'unit_price'] = np.round(df_clean.loc[other_indices, 'unit_price'], 2)
            df_clean.loc[other_indices, 'revenue'] = np.round(df_clean.loc[other_indices, 'units_sold'] * df_clean.loc[other_indices, 'unit_price'], 2)
            
            # Resolve rounding discrepancy
            cell_all_indices = df_clean[(df_clean['region'] == r) & (df_clean['quarter'] == q)].index
            target_cell_rev = target_total_revenue * quarter_shares[q] * region_shares[r]
            cell_units = df_clean.loc[cell_all_indices, 'units_sold'].values
            cell_prices = df_clean.loc[cell_all_indices, 'unit_price'].values
            target_cents = int(round(target_cell_rev * 100))
            current_cents = int(round(np.sum(cell_units * cell_prices) * 100))
            diff_cents = target_cents - current_cents
            
            if diff_cents != 0:
                step = 1 if diff_cents > 0 else -1
                remaining = abs(diff_cents)
                sorted_other_indices = other_indices[np.argsort(df_clean.loc[other_indices, 'units_sold'].values)]
                for o_idx in sorted_other_indices:
                    u = df_clean.loc[o_idx, 'units_sold']
                    if u <= remaining:
                        df_clean.loc[o_idx, 'unit_price'] += step * 0.01
                        remaining -= u
                    if remaining == 0:
                        break
                if remaining > 0:
                    o_idx = sorted_other_indices[0]
                    df_clean.loc[o_idx, 'unit_price'] += step * (remaining / 100.0) / df_clean.loc[o_idx, 'units_sold']
                    df_clean.loc[o_idx, 'unit_price'] = round(df_clean.loc[o_idx, 'unit_price'], 4)
                
                df_clean.loc[cell_all_indices, 'revenue'] = np.round(df_clean.loc[cell_all_indices, 'units_sold'] * df_clean.loc[cell_all_indices, 'unit_price'], 2)
    
    # NOW copy df_raw from df_clean AFTER all scaling and adjustments are finished!
    df_raw = df_clean.copy()
    
    # Inject the NaNs into df_raw
    df_raw.loc[chosen_region_idx, 'region'] = np.nan
    df_raw.loc[chosen_cat_idx, 'product_category'] = np.nan
    df_raw.loc[chosen_units_idx, 'units_sold'] = np.nan
    df_raw.loc[chosen_rev_idx, 'revenue'] = np.nan
    
    print(f"Injected {df_raw.isna().sum().sum()} missing values successfully.")
    
    # -------------------------------------------------------------
    # Inject Outliers (exactly 180 records)
    # -------------------------------------------------------------
    print("Generating and injecting 180 outliers...")
    outlier_rows = []
    for i in range(outliers_count):
        idx = clean_rows + i
        oid = f"ORD-{10001+idx}"
        odate = datetime.date(2025, 1, 1) + datetime.timedelta(days=int(np.random.randint(0, 365)))
        ocat = np.random.choice(categories)
        oreg = np.random.choice(regions)
        oseg = np.random.choice(segments)
        oq = f"Q{(odate.month - 1) // 3 + 1}"
        
        u_sold = int(np.random.randint(500, 1000))
        u_price = float(np.round(np.random.uniform(100.0, 200.0), 2))
        rev = float(np.round(u_sold * u_price, 2))
        
        outlier_rows.append({
            'order_id': oid,
            'order_date': odate,
            'product_category': ocat,
            'region': oreg,
            'quarter': oq,
            'units_sold': u_sold,
            'unit_price': u_price,
            'revenue': rev,
            'customer_segment': oseg
        })
    df_outliers = pd.DataFrame(outlier_rows)
    
    # -------------------------------------------------------------
    # Inject Structural Errors (exactly 120 records)
    # -------------------------------------------------------------
    print("Generating and injecting 120 structural errors (60 duplicates, 60 negative values)...")
    
    # 60 Duplicates: Select 60 rows from df_raw (which contains NaNs) so they are identical in every way
    duplicate_indices = np.random.choice(df_raw.index, size=60, replace=False)
    df_duplicates = df_raw.loc[duplicate_indices].copy()
    
    # 60 Negatives: Generate rows with negative price and revenue
    negative_rows = []
    for i in range(negatives_count):
        idx = clean_rows + outliers_count + duplicate_indices.shape[0] + i
        oid = f"ORD-{10001+idx}"
        odate = datetime.date(2025, 1, 1) + datetime.timedelta(days=int(np.random.randint(0, 365)))
        ocat = np.random.choice(categories)
        oreg = np.random.choice(regions)
        oseg = np.random.choice(segments)
        oq = f"Q{(odate.month - 1) // 3 + 1}"
        
        u_sold = int(np.random.randint(1, 10))
        u_price = float(-15.0)
        rev = float(round(u_sold * u_price, 2))
        
        negative_rows.append({
            'order_id': oid,
            'order_date': odate,
            'product_category': ocat,
            'region': oreg,
            'quarter': oq,
            'units_sold': u_sold,
            'unit_price': u_price,
            'revenue': rev,
            'customer_segment': oseg
        })
    df_negatives = pd.DataFrame(negative_rows)
    
    # Assemble raw dataset (exactly 10,000 rows)
    df_raw_final = pd.concat([df_raw, df_outliers, df_duplicates, df_negatives], ignore_index=True)
    
    # Verify the count
    print(f"Total Rows Generated: {len(df_raw_final)}")
    assert len(df_raw_final) == total_rows, f"Generated {len(df_raw_final)} rows, expected {total_rows}!"
    
    # Write to CSV
    os.makedirs("d:/data_analsyt/Retail Sales Data Analysis Dashboard", exist_ok=True)
    df_raw_final.to_csv("d:/data_analsyt/Retail Sales Data Analysis Dashboard/retail_sales_raw.csv", index=False)
    print("Successfully exported raw dataset to retail_sales_raw.csv")
    
if __name__ == "__main__":
    generate_dataset()
