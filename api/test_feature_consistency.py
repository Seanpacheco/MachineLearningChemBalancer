import pandas as pd
import joblib
import numpy as np
from feature_builder import prepare_features

# ==== Load artifacts and data ====
model = joblib.load('model/chem_balancer_ml_model.pkl')
ohe = joblib.load('model/chem_balancer_encoder.pkl')
feat_names = joblib.load('model/chem_balancer_featnames.pkl')
df = pd.read_csv('pool_chemistry_data.csv')


def test_feature_count_and_order():

    row = df.sample(1).iloc[0]
    X = prepare_features(
        data_point=row.to_dict(),
        param=row['parameter'],
        target=row['target'],
        chemical=row['chemical'],
        ohe=ohe
    )
    assert X.shape[0] == len(feat_names), \
        f"Shape mismatch: prepare_features created {X.shape} features, expected {len(feat_names)}."
    print(f"✅ Feature count matches: {len(feat_names)}")


def test_feature_order_matches():
    row = df.sample(1).iloc[0]
    feature_vector = prepare_features(
        data_point=row.to_dict(),
        param=row['parameter'],
        target=row['target'],
        chemical=row['chemical'],
        ohe=ohe
    )

    numeric_part = []
    numeric_part.append(row['pool_volume'])
    numeric_part.append(np.log1p(row['pool_volume']))
    numeric_part.append(row['pool_volume'] ** 2)  # volume_squared
    numeric_part.append(np.sqrt(row['pool_volume']))  # sqrt_volume
    numeric_part.append(row['pool_volume'] * row['target'])  # pool_vol_x_target

    numeric_part.append(row['ph'])
    numeric_part.append(row['alkalinity'])
    numeric_part.append(row['chlorine'])
    numeric_part.append(row['calcium_hardness'])
    numeric_part.append(row['cyanuric_acid'])

    numeric_part.append(row['target'])
    numeric_part.append(row['target'] - row[row['parameter']])  # delta_param
    numeric_part.append(row['ph'] * row['alkalinity'])  # ph_x_alkalinity

    is_dual_effect = 1 if row['chemical'].strip().lower() in {"sodium carbonate", "muriatic acid"} else 0
    numeric_part.append(is_dual_effect)

    if row['chemical'] == 'Muriatic Acid':
        ph_delta = abs(row['ph'] - 7.4)
        alk_delta = abs(row['alkalinity'] - 100)
        dual_intensity = ph_delta + (alk_delta / 10)
        primary_adjustment = 1 if row['parameter'] == 'alkalinity' else 0
    else:
        dual_intensity = 0
        primary_adjustment = 0

    numeric_part.append(dual_intensity)
    numeric_part.append(primary_adjustment)

    cat_array = ohe.transform([[row['chemical'], row['parameter']]])[0]

    manual_vector = np.hstack([numeric_part, cat_array])

    assert np.allclose(feature_vector, manual_vector), \
        "Feature values/order mismatch between prepare_features and manual build from training order."

    print("✅ Feature order and values match training configuration.")


if __name__ == "__main__":
    test_feature_count_and_order()
    test_feature_order_matches()
    print("All feature preparation consistency tests passed.")
