import os
import pandas as pd
import numpy as np

# Ensure seed for reproducibility
np.random.seed(42)

def make_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def generate_sales_dataset(path: str):
    print("Generating Sales Dataset...")
    rows = 1050
    dates = pd.date_range(start="2025-01-01", periods=rows, freq="h") # Date column
    
    regions = ["North", "South", "East", "West"]
    categories = ["Office Supplies", "Technology", "Furniture"]
    channels = ["Retail", "Online", "Wholesale"]
    
    data = {
        "Order ID": [f"SLS_{10000 + i}" for i in range(rows)],
        "Order Date": dates,
        "Region": np.random.choice(regions, size=rows, p=[0.4, 0.3, 0.2, 0.1]), # concentrated in North
        "Product Category": np.random.choice(categories, size=rows, p=[0.5, 0.3, 0.2]),
        "Sales Channel": np.random.choice(channels, size=rows, p=[0.6, 0.3, 0.1]),
        "Sales (USD)": np.random.exponential(scale=150.0, size=rows) + 5.0, # right skewed
        "Quantity": np.random.randint(1, 15, size=rows),
        "Discount (Fraction)": np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2], size=rows, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # 1. Inject missing values
    df.loc[df.sample(25).index, "Sales (USD)"] = np.nan
    df.loc[df.sample(20).index, "Region"] = np.nan
    df.loc[df.sample(10).index, "Order Date"] = np.nan
    df.loc[df.sample(15).index, "Discount (Fraction)"] = np.nan
    
    # 2. Inject duplicate rows (15 rows)
    dup_indices = df.sample(15).index
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 3. Inject outliers (high values)
    outlier_indices = df.sample(10).index
    df.loc[outlier_indices, "Sales (USD)"] = [9500.0, 8400.0, 12000.0, 7200.0, 11000.0, 8900.0, 15000.0, 7800.0, 9900.0, 13000.0]
    
    df.to_csv(os.path.join(path, "sales_dataset.csv"), index=False)
    print(f"Saved Sales dataset of shape {df.shape} to {path}")

def generate_hr_dataset(path: str):
    print("Generating HR Analytics Dataset...")
    rows = 1020
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="12h")
    
    depts = ["Sales", "Engineering", "HR", "Administration"]
    levels = ["Junior", "Mid", "Senior", "Director"]
    attrition = ["No", "Yes"]
    
    data = {
        "Employee ID": [f"EMP_{20000 + i}" for i in range(rows)],
        "Hire Date": dates,
        "Department": np.random.choice(depts, size=rows, p=[0.35, 0.45, 0.10, 0.10]),
        "Job Level": np.random.choice(levels, size=rows, p=[0.50, 0.30, 0.15, 0.05]),
        "Attrition": np.random.choice(attrition, size=rows, p=[0.88, 0.12]),
        "Monthly Salary": np.random.normal(loc=5500, scale=1200, size=rows), # symmetric
        "Performance Score": np.random.choice([1, 2, 3, 4, 5], size=rows, p=[0.05, 0.10, 0.60, 0.20, 0.05]),
        "Training Hours": np.random.randint(10, 120, size=rows)
    }
    
    df = pd.DataFrame(data)
    
    # 1. Inject missing values
    df.loc[df.sample(20).index, "Monthly Salary"] = np.nan
    df.loc[df.sample(15).index, "Job Level"] = np.nan
    
    # 2. Inject duplicate rows (12 rows)
    dup_indices = df.sample(12).index
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 3. Inject outliers (extreme high salaries)
    outlier_indices = df.sample(8).index
    df.loc[outlier_indices, "Monthly Salary"] = [95000.0, 85000.0, 110000.0, 75000.0, 90000.0, 82000.0, 105000.0, 80000.0]
    
    df.to_csv(os.path.join(path, "hr_dataset.csv"), index=False)
    print(f"Saved HR dataset of shape {df.shape} to {path}")

def generate_ecommerce_dataset(path: str):
    print("Generating E-commerce Dataset...")
    rows = 1100
    dates = pd.date_range(start="2026-03-01", periods=rows, freq="20min")
    
    devices = ["Mobile", "Desktop", "Tablet"]
    categories = ["Apparel", "Home Goods", "Electronics", "Books"]
    payments = ["Credit Card", "Debit Card", "Paypal", "Cash on Delivery"]
    
    data = {
        "Transaction ID": [f"TRN_{30000 + i}" for i in range(rows)],
        "Purchase Timestamp": dates,
        "Device Type": np.random.choice(devices, size=rows, p=[0.65, 0.25, 0.10]),
        "Product Category": np.random.choice(categories, size=rows, p=[0.40, 0.30, 0.20, 0.10]),
        "Payment Method": np.random.choice(payments, size=rows, p=[0.50, 0.20, 0.20, 0.10]),
        "Purchase Amount (USD)": np.random.exponential(scale=80.0, size=rows) + 2.50,
        "Items Count": np.random.randint(1, 8, size=rows),
        "Delivery Fee": np.random.uniform(0.0, 25.0, size=rows)
    }
    
    df = pd.DataFrame(data)
    
    # 1. Inject missing values
    df.loc[df.sample(30).index, "Purchase Amount (USD)"] = np.nan
    df.loc[df.sample(15).index, "Payment Method"] = np.nan
    
    # 2. Inject duplicate rows (18 rows)
    dup_indices = df.sample(18).index
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 3. Inject outliers (massive orders)
    outlier_indices = df.sample(12).index
    df.loc[outlier_indices, "Purchase Amount (USD)"] = [5200.0, 4800.0, 6500.0, 3900.0, 5800.0, 4200.0, 7100.0, 3800.0, 4900.0, 6100.0, 5500.0, 4600.0]
    
    df.to_csv(os.path.join(path, "ecommerce_dataset.csv"), index=False)
    print(f"Saved E-commerce dataset of shape {df.shape} to {path}")

def generate_banking_dataset(path: str):
    print("Generating Banking Dataset...")
    rows = 1030
    dates = pd.date_range(start="2026-05-15", periods=rows, freq="30min")
    
    types = ["Deposit", "Withdrawal", "Transfer", "Bill Pay"]
    accounts = ["Checking", "Savings", "Credit Card", "Certificate of Deposit"]
    status = ["Approved", "Declined", "Pending"]
    
    data = {
        "Transaction ID": [f"TXN_{40000 + i}" for i in range(rows)],
        "Transaction Time": dates,
        "Transaction Type": np.random.choice(types, size=rows, p=[0.45, 0.35, 0.15, 0.05]),
        "Account Type": np.random.choice(accounts, size=rows, p=[0.60, 0.25, 0.10, 0.05]),
        "Status": np.random.choice(status, size=rows, p=[0.92, 0.06, 0.02]),
        "Amount (USD)": np.random.exponential(scale=350.0, size=rows) + 1.0,
        "Account Balance (USD)": np.random.uniform(500.0, 150000.0, size=rows),
        "Process Time (ms)": np.random.normal(loc=120, scale=40, size=rows)
    }
    
    df = pd.DataFrame(data)
    
    # 1. Inject missing values
    df.loc[df.sample(25).index, "Amount (USD)"] = np.nan
    df.loc[df.sample(15).index, "Status"] = np.nan
    
    # 2. Inject duplicate rows (10 rows)
    dup_indices = df.sample(10).index
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 3. Inject outliers (huge wire transfers)
    outlier_indices = df.sample(8).index
    df.loc[outlier_indices, "Amount (USD)"] = [120000.0, 95000.0, 150000.0, 85000.0, 110000.0, 98000.0, 135000.0, 92000.0]
    
    df.to_csv(os.path.join(path, "banking_dataset.csv"), index=False)
    print(f"Saved Banking dataset of shape {df.shape} to {path}")

if __name__ == "__main__":
    output_path = os.path.join("data", "samples")
    make_directory(output_path)
    
    generate_sales_dataset(output_path)
    generate_hr_dataset(output_path)
    generate_ecommerce_dataset(output_path)
    generate_banking_dataset(output_path)
    print("Done generating test datasets pack!")
