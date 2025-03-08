# Raj Badial
# 300173931 | 09-23-24 | CSI4106 A1

import pandas as pd
url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/07/train.csv'
df = pd.read_csv(url)
print(df.head())

# Check for missing values per column
""" df.dropna(inplace=True)
missing_values = df.isnull().sum()


# Display the number of missing values for each column
print(missing_values) """

# Display unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("\n")

# Step 1: Remove '-' and '_' from values in the columns
df['Age'] = df['Age'].str.replace('-', '').str.replace('_', '')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('-', '').str.replace('_', '')
df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('-', '').str.replace('_', '')
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('-', '').str.replace('_', '')

# Step 2: Convert the cleaned columns to integers and calculate averages
df['Age'] = df['Age'].astype(int)
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(float)
df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype(float)
invested_mean = df['Amount_invested_monthly'].mean()
balance_mean = df['Monthly_Balance'].mean()
mean_numpayments = df['Num_of_Delayed_Payment'].mean()
monthly_inhand_mean = df['Monthly_Inhand_Salary'].mean()
num_credit_mean = df['Num_Credit_Inquiries'].mean()
credit_age_mode = df['Credit_History_Age'].mode()[0]

# Adjust the columns as needed
df = df[(df['Age'] >= 16) & (df['Age'] <= 120)]
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].fillna(mean_numpayments)
df['Monthly_Balance'] = df['Monthly_Balance'].fillna(balance_mean)
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].fillna(invested_mean)
df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Not Specified')
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(monthly_inhand_mean)
df['Credit_History_Age'] = df['Credit_History_Age'].fillna(credit_age_mode)
df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].fillna(num_credit_mean)

# Create a mapping of Customer_ID to Name (excluding rows where Name is missing)
customer_name_mapping = df.groupby('Customer_ID')['Name'].first().to_dict()

# Step 2: Use the mapping to fill missing names based on Customer_ID
df['Name'] = df['Name'].fillna(df['Customer_ID'].map(customer_name_mapping))


print(df.describe(include='all'))
print(df.isnull().sum())
print(df['Num_Credit_Inquiries'].describe())
#df.dropna(inplace=True)
