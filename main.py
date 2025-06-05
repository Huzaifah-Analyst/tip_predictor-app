import pandas as pd

# Load the dataset
data = pd.read_csv('tips.csv')

# Check for missing values in total_bill and tip
missing_values = data[['total_bill', 'tip']].isnull().sum()
print("Missing Values:")
print(missing_values)

# Compute summary statistics for total_bill and tip before cleaning
summary_stats = data[['total_bill', 'tip']].describe()
print("\nSummary Statistics (Before Cleaning):")
print(summary_stats)

# IQR-based outlier detection for total_bill and tip
tb_Q1 = data['total_bill'].quantile(0.25)
tb_Q3 = data['total_bill'].quantile(0.75)
tb_IQR = tb_Q3 - tb_Q1
tb_lower_bound = tb_Q1 - 1.5 * tb_IQR
tb_upper_bound = tb_Q3 + 1.5 * tb_IQR

tip_Q1 = data['tip'].quantile(0.25)
tip_Q3 = data['tip'].quantile(0.75)
tip_IQR = tip_Q3 - tip_Q1
tip_lower_bound = tip_Q1 - 1.5 * tip_IQR
tip_upper_bound = tip_Q3 + 1.5 * tip_IQR

# Remove IQR-based outliers
cleaned_data = data[(data['total_bill'] >= tb_lower_bound) & (data['total_bill'] <= tb_upper_bound) & 
                    (data['tip'] >= tip_lower_bound) & (data['tip'] <= tip_upper_bound)]
print("\nShape of Data Before Cleaning:", data.shape)
print("Shape of Data After IQR Cleaning:", cleaned_data.shape)

# Remove low-frequency bins (bills >= 40)
cleaned_data = cleaned_data[cleaned_data['total_bill'] < 40]
print("Shape of Data After Removing Low-Frequency Bins (< 40):", cleaned_data.shape)

# Convert total_bill and tip to integers (rounding to nearest whole number)
cleaned_data['total_bill'] = cleaned_data['total_bill'].round().astype(int)
cleaned_data['tip'] = cleaned_data['tip'].round().astype(int)

# Summary statistics after all cleaning
final_summary = cleaned_data[['total_bill', 'tip']].describe()
print("\nSummary Statistics (After All Cleaning and Conversion to Integers):")
print(final_summary)

# Distribution of total_bill after all cleaning
bins = [0, 10, 20, 30, 40]
labels = ['0-10', '10-20', '20-30', '30-40']
bill_dist = pd.cut(cleaned_data['total_bill'], bins=bins, labels=labels, right=False).value_counts().sort_index()
print("\nDistribution of Total Bill After All Cleaning:")
print(bill_dist)

# Save final cleaned data
cleaned_data.to_csv('final_cleaned_tips.csv', index=False)
print("\nFinal cleaned data saved to 'final_cleaned_tips.csv'")