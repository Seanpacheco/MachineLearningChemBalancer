import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from feature_builder import prepare_features, dual_effect_chems, param_map

class EnsembleModel:
    def __init__(self, models, weights):
        self.models = models  # List of trained models
        self.weights = weights  # List of weights for each model
    
    def predict(self, X):
        """Make weighted ensemble prediction"""
        preds = [model.predict(X) * weight for model, weight in zip(self.models, self.weights)]
        return sum(preds)

df = pd.read_csv('pool_chemistry_data.csv')
print(f"Loaded {len(df)} samples")

def robust_outlier_treatment(df):
    cleaned_df = pd.DataFrame()
    for chemical in df['chemical'].unique():
        chem_data = df[df['chemical'] == chemical].copy()
        if chemical in ['Muriatic Acid', 'Calcium Chloride (77%)', 'Sodium Bicarbonate']:
            Q1 = chem_data['dosage'].quantile(0.01)
            Q3 = chem_data['dosage'].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
        else:
            Q1 = chem_data['dosage'].quantile(0.05)
            Q3 = chem_data['dosage'].quantile(0.95)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
        clean_chem_data = chem_data[
            (chem_data['dosage'] >= lower_bound) & 
            (chem_data['dosage'] <= upper_bound)
        ]
        cleaned_df = pd.concat([cleaned_df, clean_chem_data], ignore_index=True)
    return cleaned_df

df = robust_outlier_treatment(df)
print(f"Cleaned data size: {len(df)}")

X_list = []
y_list = []

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_data = df[['chemical', 'parameter']].copy()
cat_data['chem_param'] = df['chemical'] + '_' + df['parameter']
ohe.fit(cat_data)

for _, row in df.iterrows():
    try:
        features = prepare_features(
            data_point=row.to_dict(),
            param=row['parameter'],
            target=row['target'],
            chemical=row['chemical'],
            ohe=ohe
        )
        X_list.append(features)
        y_list.append(row['dosage'])
    except Exception as e:
        continue

X = np.array(X_list)
y = np.array(y_list)
y_log = np.log1p(y)

def calculate_extreme_weights(dosages):
    """Aggressive weighting for extreme dosage cases"""
    weights = np.ones(len(dosages))
    
    weights[dosages > 200] = 3.0   # 3x weight for high dosage
    weights[dosages > 500] = 5.0   # 5x weight for very high dosage
    weights[dosages > 1000] = 8.0  # 8x weight for extreme dosage
    weights[dosages > 2000] = 12.0 # 12x weight for ultra-extreme
    
    return weights

sample_weights = calculate_extreme_weights(y)
print(f"Extreme dosage samples (>1000 oz): {sum(y > 1000)}")
print(f"Ultra-extreme samples (>2000 oz): {sum(y > 2000)}")

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y_log, sample_weights, test_size=0.2, random_state=42
)

main_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.02,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1
)

main_model.fit(X_train, y_train, sample_weight=weights_train)

numeric_features = [
    'current_val', 'target', 'ppm_change', 'pool_volume', 'log_pool_volume',
    'volume_scaled_change', 'chemical_multiplier', 'current_x_volume_x_target',
    'current_x_volume_x_delta', 'volume_x_delta', 'change_ratio',
    'concentration_factor', 'buffer_strength', 'is_dual_effect',
    'ph', 'alkalinity', 'chlorine', 'calcium_hardness', 'cyanuric_acid'
]

from itertools import chain
try:
    cat_feature_names = list(chain.from_iterable(
        [f'{col}_{cat}' for cat in cats] for col, cats in zip(['chemical', 'parameter', 'chem_param'], ohe.categories_)
    ))
except Exception:
    cat_feature_names = []
    print("Could not extract OHE feature names. They may differ by sklearn version.")

all_feature_names = numeric_features + list(cat_feature_names)

importances = main_model.feature_importances_

import numpy as np
top_indices = np.argsort(importances)[::-1][:15]
print("\nTop 15 Most Important Features (main model):")
for idx in top_indices:
    fname = all_feature_names[idx] if idx < len(all_feature_names) else f'feature_{idx}'
    print(f"{fname:40s} Importance: {importances[idx]:.4f}")

def create_extreme_ensemble(df, chemical, ohe):
    """Create ensemble of specialized models for extreme cases"""
    print(f"Creating extreme ensemble for {chemical}...")
    
    chem_data = df[df['chemical'] == chemical].copy()
    if len(chem_data) < 100:
        return None
    
    X_chem_list = []
    y_chem_list = []
    
    for _, row in chem_data.iterrows():
        try:
            features = prepare_features(
                data_point=row.to_dict(),
                param=row['parameter'],
                target=row['target'],
                chemical=row['chemical'],
                ohe=ohe
            )
            X_chem_list.append(features)
            y_chem_list.append(row['dosage'])
        except:
            continue
    
    if len(X_chem_list) < 50:
        return None
    
    X_chem = np.array(X_chem_list)
    y_chem = np.log1p(np.array(y_chem_list))
    
    X_train_chem, X_test_chem, y_train_chem, y_test_chem = train_test_split(
        X_chem, y_chem, test_size=0.2, random_state=42
    )
    
    ensemble_models = []
    weights = []
    
    xgb_extreme = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=12,
        learning_rate=0.01,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    xgb_extreme.fit(X_train_chem, y_train_chem)
    ensemble_models.append(xgb_extreme)
    weights.append(0.5)
    
    from sklearn.ensemble import RandomForestRegressor
    rf_extreme = RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_extreme.fit(X_train_chem, y_train_chem)
    ensemble_models.append(rf_extreme)
    weights.append(0.3)
    
    qr_extreme = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.5,  # Median
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        random_state=42
    )
    qr_extreme.fit(X_train_chem, y_train_chem)
    ensemble_models.append(qr_extreme)
    weights.append(0.2)
    

    for model, name in zip(ensemble_models, ['xgb_extreme', 'rf_extreme', 'qr_extreme']):
        y_pred = model.predict(X_test_chem)
        mae = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_test_chem)))
        print(f"  {name} MAE: {mae:.3f} oz")

    return EnsembleModel(ensemble_models, weights)


extreme_ensembles = {}
high_dosage_chemicals = ['Muriatic Acid', 'Calcium Chloride (77%)', 'Sodium Bicarbonate']

for chemical in high_dosage_chemicals:
    ensemble_func = create_extreme_ensemble(df, chemical, ohe)
    if ensemble_func:
        extreme_ensembles[chemical] = ensemble_func

print(f"Created {len(extreme_ensembles)} extreme ensembles")

def multi_stage_prediction(data_point, param, target, chemical, main_model, extreme_ensembles, ohe):
    """
    Multi-stage pipeline that routes predictions based on dosage level and chemical complexity
    """
    X = prepare_features(data_point, param, target, chemical, ohe)
    base_log_pred = main_model.predict([X])[0]
    base_pred = np.expm1(base_log_pred)
    
    if chemical in extreme_ensembles:
        if base_pred > 1000: 
            ensemble_log_pred = extreme_ensembles[chemical]([X])[0]
            ensemble_pred = np.expm1(ensemble_log_pred)
            final_pred = 0.3 * base_pred + 0.7 * ensemble_pred
            return final_pred, "extreme_ensemble"
            
        elif base_pred > 500:  
            ensemble_log_pred = extreme_ensembles[chemical]([X])[0]
            ensemble_pred = np.expm1(ensemble_log_pred)
            final_pred = 0.5 * base_pred + 0.5 * ensemble_pred
            return final_pred, "ensemble_blend"
    
    if chemical == 'Muriatic Acid' and base_pred > 200:
        conservative_pred = base_pred * 0.9
        return conservative_pred, "conservative_main"
    
    return base_pred, "main_model"

correct_predictions = 0
total_extreme_cases = 0

for _, row in df[df['dosage'] > 1000].sample(100).iterrows():
    try:
        pred, model_used = multi_stage_prediction(
            row.to_dict(), row['parameter'], row['target'], 
            row['chemical'], main_model, extreme_ensembles, ohe
        )
        actual = row['dosage']
        error = abs(pred - actual)
        
        if error < actual * 0.3:
            correct_predictions += 1
        total_extreme_cases += 1
        
    except:
        continue

print(f"Extreme case accuracy: {correct_predictions}/{total_extreme_cases} = {correct_predictions/max(total_extreme_cases,1)*100:.1f}%")

joblib.dump(main_model, 'model/chem_balancer_ml_model.pkl')
joblib.dump(ohe, 'model/chem_balancer_encoder.pkl')
joblib.dump(extreme_ensembles, 'model/extreme_ensembles.pkl')
joblib.dump(True, 'model/use_log_transform.pkl')

print("💾 All models saved successfully.")
