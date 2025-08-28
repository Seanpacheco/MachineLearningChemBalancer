# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the pool chemistry dataset
df = pd.read_csv('sample-data.csv')
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
df.head()

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
df.describe()

# 1. Distribution Analysis - Water Parameter Histograms
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
parameters = ['ph', 'alkalinity', 'chlorine', 'calcium_hardness', 'cyanuric_acid', 'pool_volume']

for i, param in enumerate(parameters):
    row = i // 2
    col = i % 2
    axes[row, col].hist(df[param], bins=20, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'Distribution of {param.title()}')
    axes[row, col].set_xlabel(param.title())
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Pool Chemistry Parameter Distributions', fontsize=16, y=1.02)
plt.show()

# 2. Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[parameters].corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.3f')
plt.title('Correlation Matrix of Pool Water Parameters', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 3. Chemical Usage Analysis - Dosage by Chemical Type
plt.figure(figsize=(15, 8))
chemical_counts = df['chemical'].value_counts()
top_chemicals = chemical_counts.head(10).index

# Filter for top chemicals
df_top = df[df['chemical'].isin(top_chemicals)]

# Create boxplot
sns.boxplot(data=df_top, x='chemical', y='dosage')
plt.xticks(rotation=45, ha='right')
plt.title('Dosage Distribution by Chemical Type', fontsize=14)
plt.xlabel('Chemical')
plt.ylabel('Dosage')
plt.tight_layout()
plt.show()

# 4. Parameter vs Target Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
parameters_to_plot = ['ph', 'alkalinity', 'chlorine', 'calcium_hardness']

for i, param in enumerate(parameters_to_plot):
    row = i // 2
    col = i % 2
    
    # Filter data for specific parameter adjustments
    param_data = df[df['parameter'] == param]
    if not param_data.empty:
        axes[row, col].scatter(param_data['target'], param_data['dosage'], alpha=0.6)
        axes[row, col].set_title(f'{param.title()}: Dosage vs Target')
        axes[row, col].set_xlabel('Target Value')
        axes[row, col].set_ylabel('Dosage')
        axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Parameter Adjustment Analysis: Dosage vs Target Values', fontsize=16, y=1.02)
plt.show()

# 5. Pool Volume Impact Analysis
plt.figure(figsize=(12, 8))
volume_bins = pd.cut(df['pool_volume'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
sns.boxplot(x=volume_bins, y=df['dosage'])
plt.title('Chemical Dosage by Pool Volume Category', fontsize=14)
plt.xlabel('Pool Volume Category')
plt.ylabel('Dosage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Chemical Efficiency Analysis
plt.figure(figsize=(14, 6))

# Calculate dosage per unit volume
df['dosage_per_volume'] = df['dosage'] / df['pool_volume'] * 1000  # per 1000 gallons

# Top chemicals by usage
top_chems = df['chemical'].value_counts().head(8).index
df_efficiency = df[df['chemical'].isin(top_chems)]

sns.boxplot(data=df_efficiency, x='chemical', y='dosage_per_volume')
plt.xticks(rotation=45, ha='right')
plt.title('Chemical Efficiency: Dosage per 1000 Gallons by Chemical Type', fontsize=14)
plt.xlabel('Chemical')
plt.ylabel('Dosage per 1000 Gallons')
plt.tight_layout()
plt.show()

# 7. Machine Learning Model Performance Visualization
# Prepare data for modeling
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le_chemical = LabelEncoder()
le_parameter = LabelEncoder()

df_model = df.copy()
df_model['chemical_encoded'] = le_chemical.fit_transform(df['chemical'])
df_model['parameter_encoded'] = le_parameter.fit_transform(df['parameter'])

# Features and target
features = ['ph', 'alkalinity', 'chlorine', 'calcium_hardness', 'cyanuric_acid', 
           'pool_volume', 'target', 'chemical_encoded', 'parameter_encoded']
X = df_model[features]
y = np.log1p(df_model['dosage'])  # Log transform for better modeling

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model performance visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes.set_xlabel('Actual (log dosage)')
axes.set_ylabel('Predicted (log dosage)')
axes.set_title('Actual vs Predicted Dosage')
axes.grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted (log dosage)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

axes[2].barh(feature_importance['feature'], feature_importance['importance'])
axes[2].set_xlabel('Feature Importance')
axes[2].set_title('Feature Importance in ML Model')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print model metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Square Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 8. Interactive Dose Prediction Interface
def predict_dosage(ph, alkalinity, chlorine, calcium_hardness, cyanuric_acid, 
                  pool_volume, target_param, chemical_name):
    """
    Interactive function to predict chemical dosage
    """
    try:
        # Encode inputs
        chemical_encoded = le_chemical.transform([chemical_name])[0] if chemical_name in le_chemical.classes_ else 0
        parameter_encoded = le_parameter.transform([target_param]) if target_param in le_parameter.classes_ else 0
        
        # Create input array
        input_features = np.array([[ph, alkalinity, chlorine, calcium_hardness, 
                                  cyanuric_acid, pool_volume, 7.5,  # default target
                                  chemical_encoded, parameter_encoded]])
        
        # Predict
        log_dosage = model.predict(input_features)[0]
        dosage = np.expm1(log_dosage)  # Reverse log transform
        
        return dosage
    except:
        return "Error in prediction - check input values"

# Example usage
print("Example Dosage Predictions:")
print("=" * 50)

# Test scenarios
test_cases = [
    {"ph": 7.4, "alkalinity": 80, "chlorine": 1, "calcium_hardness": 200, 
     "cyanuric_acid": 30, "pool_volume": 10000, "target_param": "chlorine", 
     "chemical_name": "Chlorine Gas"},
    
    {"ph": 7.0, "alkalinity": 60, "chlorine": 2, "calcium_hardness": 150, 
     "cyanuric_acid": 25, "pool_volume": 5000, "target_param": "alkalinity", 
     "chemical_name": "Sodium Bicarbonate"}
]

for i, case in enumerate(test_cases, 1):
    result = predict_dosage(**case)
    print(f"Test Case {i}:")
    print(f"  Input: {case}")
    print(f"  Predicted Dosage: {result:.3f} units")
    print()

# 9. Summary Statistics and Data Quality Check
print("Pool Chemistry Dataset Summary")
print("=" * 50)
print(f"Total Records: {len(df):,}")
print(f"Unique Chemicals: {df['chemical'].nunique()}")
print(f"Unique Parameters: {df['parameter'].nunique()}")
print(f"Pool Volume Range: {df['pool_volume'].min():,} - {df['pool_volume'].max():,} gallons")
print(f"Dosage Range: {df['dosage'].min():.3f} - {df['dosage'].max():.3f}")

# Missing values check
print("\nMissing Values:")
print(df.isnull().sum())

# Chemical distribution
print("\nTop 10 Chemicals by Usage:")
print(df['chemical'].value_counts().head(10))

# Parameter distribution
print("\nParameter Distribution:")
print(df['parameter'].value_counts())
