# # -------------------------- part 1 --------------------------

# import pandas as pd

# df=pd.read_csv('loans.csv')
# print(100*"-")

# print(df.info())
# print(100*"-")

# print(df.head())
# print(100*"-")

# # -------------------------- part 2 --------------------------

# import pandas as pd

# df=pd.read_csv('loans.csv')

# missing_values=df.isnull().sum()
# print("Missing Values :\n", missing_values)

# df_dropped_rows=df.dropna()   # Delete row
# df_dropped_columns=df.dropna(axis=1)    # Delete column

# # replacement
# df['loan_amount'].fillna(df['loan_amount'].mean(),inplace=True)
# df['loan_type'].fillna(df['loan_type'].mode()[0],inplace=True)

# # -------------------------- part 3 --------------------------

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df=pd.read_csv('loans.csv')

# plt.figure(figsize=(8,6))
# sns.boxplot(x=df['loan_amount'])
# plt.title('Plot of Loan Amount')
# plt.show()

# plt.figure(figsize=(8,6))
# sns.boxplot(x=df['rate'])
# plt.title('Plot of Rate')
# plt.show()


# # mohasebe Q1, Q3, and IQR for 'loan_amount'
# Q1_amount=df['loan_amount'].quantile(0.25)
# Q3_amount=df['loan_amount'].quantile(0.75)
# IQR_amount=Q3_amount-Q1_amount

# # yaftane mahdode upper and lower 
# lower_bound_amount=Q1_amount-1.5*IQR_amount
# upper_bound_amount=Q3_amount+1.5*IQR_amount

# # Filter data part
# df_filtered=df[(df['loan_amount']>=lower_bound_amount)&(df['loan_amount']<=upper_bound_amount)]

# # mohasebe Q1, Q3, and IQR for 'rate'
# Q1_rate=df['rate'].quantile(0.25)
# Q3_rate=df['rate'].quantile(0.75)
# IQR_rate=Q3_rate-Q1_rate

# # yaftane mahdode upper and lower 
# lower_bound_rate=Q1_rate-1.5*IQR_rate
# upper_bound_rate=Q3_rate+1.5*IQR_rate

# # Filter data part
# df_filtered=df_filtered[(df_filtered['rate']>=lower_bound_rate)&(df_filtered['rate']<=upper_bound_rate)]

# # -------------------------- part 4 --------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# df=pd.read_csv('loans.csv')

# Q1_amount=df['loan_amount'].quantile(0.25)
# Q3_amount=df['loan_amount'].quantile(0.75)
# IQR_amount=Q3_amount-Q1_amount
# lower_bound_amount=Q1_amount-1.5*IQR_amount
# upper_bound_amount=Q3_amount+1.5*IQR_amount

# # mahdodsazi
# df['loan_amount']=df['loan_amount'].apply(lambda x:upper_bound_amount if x>upper_bound_amount else (lower_bound_amount if x<lower_bound_amount else x))

# # Calculate skewness before transformation
# print("Skewness before transformation:")
# print(f"Loan Amount Skewness: {df['loan_amount'].skew():.4f}")
# print(f"Rate Skewness: {df['rate'].skew():.4f}")

# # distribution before transformation
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sns.histplot(df['loan_amount'],kde=True)
# plt.title('Loan Amount Distribution (Original)')

# plt.subplot(1,2,2)
# sns.histplot(df['rate'],kde=True)
# plt.title('Rate Distribution (Original)')
# plt.show()

# # new transformed columns

# # 1. Log Transformation for 'loan_amount'
# df['loan_amount_log']=np.log(df['loan_amount'])

# # 2. Log Transformation for 'rate'
# df['rate_log']=np.log(df['rate'])

# # Calculate skewness after transformation
# print("\nSkewness after Log Transformation:")
# print(f"Loan Amount Log Skewness: {df['loan_amount_log'].skew():.4f}")
# print(f"Rate Log Skewness: {df['rate_log'].skew():.4f}")

# # distribution after transformation
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# sns.histplot(df['loan_amount_log'],kde=True)
# plt.title('Loan Amount Distribution (Log Transformed)')

# plt.subplot(1,2,2)
# sns.histplot(df['rate_log'],kde=True)
# plt.title('Rate Distribution (Log Transformed)')
# plt.show()

# # Update the main DataFrame to transformed values
# df['loan_amount']=df['loan_amount_log']
# df['rate']=df['rate_log']
# df=df.drop(columns=['loan_amount_log','rate_log'],errors='ignore')

# # -------------------------- part 5 --------------------------

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# df=pd.read_csv('loans.csv')

# numerical_features = ['loan_amount', 'rate']

# # Initialize the StandardScaler
# scaler=StandardScaler()

# # 4. Fit the scaler to the data
# df[numerical_features]=scaler.fit_transform(df[numerical_features])

# print("DataFrame Head after Standardization:")
# print(df.head())
# print("\nDescriptive Statistics after Standardization (should be close to mean=0,std=1):")
# print(df[numerical_features].describe())

# # Save the updated DataFrame
# df.to_csv('loans_scaled.csv',index=False)

# # -------------------------- part 6 --------------------------

# import pandas as pd

# df=pd.read_csv('loans.csv')

# categorical_feature=['loan_type']

# # Apply One-Hot Encoding
# df_encoded=pd.get_dummies(df,columns=categorical_feature,prefix=categorical_feature)

# print("DataFrame Head after One-Hot Encoding:")
# print(df_encoded.head())
# print("\nDataFrame Columns and Dtypes:")
# df_encoded.info()

# # Save the updated DataFrame
# df_encoded.to_csv('loans_encoded.csv',index=False)

# # -------------------------- part 7 --------------------------

# import pandas as pd

# df=pd.read_csv('loans_encoded.csv')

# # Convert date columns to datetime objects
# df['loan_start']=pd.to_datetime(df['loan_start'])
# df['loan_end']=pd.to_datetime(df['loan_end'])

# # Create the new feature
# df['loan_duration_days']=(df['loan_end']-df['loan_start']).dt.days

# print("New Feature: loan_duration_days")
# print(df[['loan_start','loan_end','loan_duration_days']].head())

# # # --

# # Create a new interaction feature
# df['loan_rate_interaction']=df['loan_amount']*df['rate']

# print("\nNew Feature: loan_rate_interaction")
# print(df[['loan_amount','rate','loan_rate_interaction']].head())

# # # --

# # hazf unnecessary columns
# columns_to_drop=['client_id','loan_id','loan_start','loan_end']
# df=df.drop(columns=columns_to_drop)

# print("\nFinal DataFrame Columns and Info:")
# df.info()

# # Save the updated DataFrame
# df.to_csv('loans_engineered.csv',index=False)

# # -------------------------- part 8 --------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('loans_engineered.csv')

# Separate Features (X) and Target (y)
X=df.drop('repaid',axis=1)     # All columns bejoz 'repaid'
y=df['repaid']                 # 'repaid' column

# 3. Split the data into Training and Test (70% train,30% test)
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

print("Data Splitting Complete:")
print(f"Total Rows: {len(df)}")
print("-"*30)
print(f"Training Set Size (X_train): {len(X_train)} rows")
print(f"Test Set Size (X_test): {len(X_test)} rows")