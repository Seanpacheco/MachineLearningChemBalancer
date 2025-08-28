import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("pool_chemistry_data.csv")

# 1. View basic info
print(df.info())
print(df.describe())

# 2. Distribution plots
plt.figure(figsize=(12,6))
for i, col in enumerate(['ph', 'alkalinity', 'chlorine', 'calcium_hardness', 'cyanuric_acid']):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# 3. Check for missing/odd values
print(df.isna().sum())
print("Any negative dosages?", (df['dosage'] < 0).sum())

# 4. Boxplots, e.g., dosage by chemical/parameter
plt.figure(figsize=(12,4))
sns.boxplot(data=df, x='chemical', y='dosage')
plt.title("Dosage by Chemical")
plt.xticks(rotation=20)
plt.show()

# 5. Correlation heatmap of numeric features
plt.figure(figsize=(8,6))
sns.heatmap(df[['ph','alkalinity','chlorine','calcium_hardness','cyanuric_acid','pool_volume','dosage']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 6. Pivot to see which chemicals are used for which adjustments
pd.crosstab(df.chemical, df.parameter)

# 7. Mean dosage needed vs pool_volume
sns.scatterplot(x='pool_volume', y='dosage', hue='chemical', data=df.sample(200))  # limit for clarity
plt.title("Dosage vs Pool Volume")
plt.show()
