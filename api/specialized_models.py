import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from feature_builder import prepare_features

df = pd.read_csv('pool_chemistry_data.csv')
ohe = joblib.load('model/chem_balancer_encoder.pkl')

high_error_chemicals = ['Muriatic Acid', 'Calcium Chloride (77%)', 'Sodium Bicarbonate']

specialized_models = {}

for chemical in high_error_chemicals:
    print(f"\nTraining specialized model for {chemical}...")
    
    chem_data = df[df['chemical'] == chemical].copy()
    
    if len(chem_data) < 50: 
        print(f"Insufficient data for {chemical}: {len(chem_data)} samples")
        continue
    
    X_chem = []
    y_chem = []
    
    for _, row in chem_data.iterrows():
        try:
            features = prepare_features(
                row.to_dict(), row['parameter'], 
                row['target'], row['chemical'], ohe
            )
            X_chem.append(features)
            y_chem.append(np.log1p(row['dosage']))
        except:
            continue
    
    X_chem = np.array(X_chem)
    y_chem = np.array(y_chem)
    
    specialized_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2,
        min_child_weight=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_chem, y_chem, test_size=0.2, random_state=42
    )
    
    specialized_model.fit(X_train, y_train)
    
    y_pred = specialized_model.predict(X_test)
    mae = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_test)))
    
    print(f"{chemical} specialized model MAE: {mae:.2f}")
    specialized_models[chemical] = specialized_model

for chemical, model in specialized_models.items():
    safe_name = chemical.replace('(', '').replace(')', '').replace('%', 'pct').replace(' ', '_')
    joblib.dump(model, f'model/specialized_{safe_name}.pkl')

print(f"\nSaved {len(specialized_models)} specialized models")
