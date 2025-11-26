import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=cols)

# Remove NaN
df.dropna(inplace=True)

# Removing constant columns and keep only columns where the number of unique values is > 1
df = df.loc[:, df.nunique() > 1]
print(f"Data shape after dropping constants: {df.shape}")

# RUL and Labeling
def calculate_rul(df):
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max']
    df = df.merge(max_cycle, on='unit', how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

df = calculate_rul(df)

# Label: 1 if RUL <= 20 (Failure), else 0 (Normal)
df['label'] = np.where(df['RUL'] <= 20, 1, 0)

# Feature Selection for correlation => 0.4
corr_matrix = df.corr()
target_corr = abs(corr_matrix['label'])

threshold = 0.4
selected_features = target_corr[target_corr >= threshold].index.tolist()

for col in ['label', 'RUL', 'unit', 'cycle']:
    if col in selected_features:
        selected_features.remove(col)

print(f"Selected Features (Corr >= {threshold}): {selected_features}")

#Save da output!!!
final_columns = selected_features + ['label']
processed_df = df[final_columns]

output_path = 'data/Processed_train_FD001.csv'
processed_df.to_csv(output_path, index=False)
joblib.dump(selected_features, 'selected_features.pkl')

print(f"All Done SIR")