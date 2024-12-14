import pandas as pd
parquet_file_path = "wrf3km_100m_combined.parquet"
df = pd.read_parquet(parquet_file_path)

print("DataFrame Overview:")
print(df.info())
print(df.describe())
print(df.head())
print(df.columns.tolist())
