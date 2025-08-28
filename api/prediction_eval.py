import pandas as pd
import numpy as np
import os
import joblib
from feature_builder import prepare_features


model = joblib.load('model/chem_balancer_ml_model.pkl')
ohe = joblib.load('model/chem_balancer_encoder.pkl')
try:
    use_log_transform = joblib.load('model/use_log_transform.pkl')
except:
    use_log_transform = False


specialized_models = {}
specialized_chemicals = ['Muriatic_Acid', 'Calcium_Chloride_77pct', 'Sodium_Bicarbonate']
for chem in specialized_chemicals:
    model_path = f'model/specialized_{chem}.pkl'
    if os.path.exists(model_path):
        specialized_models[chem.replace('_', ' ').replace('77pct', '(77%)')] = joblib.load(model_path)


try:
    extreme_ensembles = joblib.load('model/extreme_ensembles.pkl')
    print(f"Loaded {len(extreme_ensembles)} extreme ensembles")
except:
    extreme_ensembles = {}
    print("No extreme ensembles found")


def multi_stage_prediction(data_point, param, target, chemical):
    """Enhanced prediction using main model + specialized models + ensembles"""
    X = prepare_features(data_point, param, target, chemical, ohe)
    

    if use_log_transform:
        log_pred = model.predict([X])[0]
        base_pred = np.expm1(log_pred)
    else:
        base_pred = model.predict([X])[0]
    
    base_pred = max(base_pred, 0.01)
    
    if chemical in extreme_ensembles:
        if base_pred > 1000:  
            try:
                ensemble_pred_log = extreme_ensembles[chemical].predict([X])[0]
                ensemble_pred = np.expm1(ensemble_pred_log)
                return 0.3 * base_pred + 0.7 * ensemble_pred
            except:
                pass
        elif base_pred > 500: 
            try:
                ensemble_pred_log = extreme_ensembles[chemical].predict([X])[0]
                ensemble_pred = np.expm1(ensemble_pred_log)
                return 0.5 * base_pred + 0.5 * ensemble_pred
            except:
                pass
    

    if chemical in specialized_models and base_pred > 500:
        try:
            if use_log_transform:
                sp_log_pred = specialized_models[chemical].predict([X])[0]
                sp_pred = np.expm1(sp_log_pred)
            else:
                sp_pred = specialized_models[chemical].predict([X])[0]
            return 0.5 * base_pred + 0.5 * sp_pred
        except:
            pass
    

    if chemical == 'Muriatic Acid' and base_pred > 200:
        return base_pred * 0.9
    
    return base_pred


df = pd.read_csv('pool_chemistry_data.csv')
print(f"OneHotEncoder categories: {ohe.categories_}")
print(f"Model expects {model.n_features_in_} features")

predictions = []
failed_predictions = 0

for _, row in df.iterrows():
    try:
        pred = multi_stage_prediction(
            row.to_dict(), 
            row['parameter'], 
            row['target'], 
            row['chemical']
        )
        predictions.append(pred)
    except Exception as e:
        print(f"Error predicting row {row.name}: {e}")
        print(f" Chemical: {row['chemical']}, Parameter: {row['parameter']}")
        predictions.append(np.nan)
        failed_predictions += 1

print(f"Failed predictions: {failed_predictions}/{len(df)}")

df['predicted'] = predictions
df['abs_error'] = np.abs(df['predicted'] - df['dosage'])
df['relative_error'] = df['abs_error'] / df['dosage']

df_clean = df.dropna(subset=['predicted'])

summary = df_clean.groupby('chemical').agg({
    'abs_error': ['mean', 'std', 'median', 'min', 'max'],
    'relative_error': ['mean', 'std', 'median', 'min', 'max'],
    'dosage': ['count', 'mean', 'min', 'max']
}).round(4)

summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.sort_values('abs_error_mean', ascending=False)

print("Chemical Performance Summary (worst absolute error first):")
print(summary)

df_clean['dosage_range'] = pd.cut(df_clean['dosage'],
                                 bins=[0, 10, 50, 200, 1000, float('inf')],
                                 labels=['Very Low (0-10)', 'Low (10-50)', 'Medium (50-200)',
                                        'High (200-1000)', 'Very High (1000+)'])

dosage_analysis = df_clean.groupby('dosage_range').agg({
    'abs_error': ['mean', 'std'],
    'relative_error': ['mean', 'std'],
    'dosage': ['count', 'mean']
}).round(4)

dosage_analysis.columns = ['_'.join(col) for col in dosage_analysis.columns]
print("\nPerformance by Dosage Range:")
print(dosage_analysis)

df_clean['chem_param'] = df_clean['chemical'] + '_' + df_clean['parameter']
combo_analysis = df_clean.groupby('chem_param').agg({
    'abs_error': ['mean', 'count'],
    'relative_error': 'mean',
    'dosage': 'mean'
}).round(4)

combo_analysis.columns = ['_'.join(col) for col in combo_analysis.columns]
combo_analysis = combo_analysis[combo_analysis['abs_error_count'] >= 5]
combo_analysis = combo_analysis.sort_values('abs_error_mean', ascending=False)

print(f"\nWorst Chemical-Parameter Combinations (min 5 samples):")
print(combo_analysis.head(10))

print(f"\n📊 Overall Model Performance:")
print(f"Mean Absolute Error: {df_clean['abs_error'].mean():.2f}")
print(f"Mean Relative Error: {df_clean['relative_error'].mean():.4f}")
print(f"Median Absolute Error: {df_clean['abs_error'].median():.2f}")
print(f"Median Relative Error: {df_clean['relative_error'].median():.4f}")

summary.to_csv('chemical_performance_analysis.csv')
dosage_analysis.to_csv('dosage_range_analysis.csv')
combo_analysis.to_csv('chemical_parameter_combinations.csv')
