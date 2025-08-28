import pandas as pd, random
from api import raise_chemicals, lower_chemicals, chlorine_raise_options

ALK_TOL, PH_TOL, CYA_TOL, CH_TOL = 5, 0.05, 2, 5


IDEAL = {
    'ph': (7.4, 7.6),
    'alkalinity': (80, 120),
    'chlorine': (3, 5),
    'calcium_hardness': (200, 400),
    'cyanuric_acid': (30, 50)
}

RANGES = {
    'ph': (7.2, 7.8),
    'alkalinity': (60, 180),
    'chlorine': (1, 5),
    'calcium_hardness': (150, 800),
    'cyanuric_acid': (10, 80)
}

EXTREME = {
    'ph': (6.5, 8.5),
    'alkalinity': (30, 250),
    'chlorine': (0, 10),
    'calcium_hardness': (100, 1000),
    'cyanuric_acid': (0, 150)
}


IDEAL_PROB = 0.8          
BOTH_OUT_PROB = 0.6        
EXTREME_PROB = 0.05     
ALK_PRIORITY_MULTIPLIER = 2


df_base = pd.read_csv("pool_chemistry_data.csv")
grouped_by_vol = {vol: df for vol, df in df_base.groupby('pool_volume')}


def rand_weighted(param):
    """Pick value from ideal, acceptable, or extreme range."""
    rnd = random.random()
    if rnd < EXTREME_PROB: 
        lo, hi = EXTREME[param]
        if random.choice([True, False]):
            return round(random.uniform(lo, RANGES[param][0] - 0.01), 2) if isinstance(lo, float) \
                   else int(round(random.uniform(lo, RANGES[param][0] - 1)))
        else:
            return round(random.uniform(RANGES[param][1] + 0.01, hi), 2) if isinstance(hi, float) \
                   else int(round(random.uniform(RANGES[param][1] + 1, hi)))
    elif rnd < IDEAL_PROB + EXTREME_PROB:  
        lo, hi = IDEAL[param]
    else:  
        lo, hi = RANGES[param]

    return round(random.uniform(lo, hi), 2) if isinstance(lo, float) else int(round(random.uniform(lo, hi)))


def pick_chemical(parameter, current_val, target_val, available_df):
    if parameter == 'alkalinity' and abs(current_val - target_val) <= ALK_TOL: return None
    if parameter == 'ph' and abs(current_val - target_val) <= PH_TOL: return None
    if parameter == 'cyanuric_acid' and abs(current_val - target_val) <= CYA_TOL: return None
    if parameter == 'calcium_hardness' and abs(current_val - target_val) <= CH_TOL: return None

    if target_val > current_val:
        if parameter == 'chlorine':
            return next((c for c in chlorine_raise_options if c in available_df['chemical'].values), None)
        chem_opt = raise_chemicals.get(parameter)
        if isinstance(chem_opt, list):
            opts = [c for c in chem_opt if c in available_df['chemical'].values]
            return random.choice(opts) if opts else None
        return chem_opt if chem_opt in available_df['chemical'].values else None
    else:
        if parameter == 'chlorine':
            cname = lower_chemicals.get('chlorine')
            return cname if cname in available_df['chemical'].values else None
        cname = lower_chemicals.get(parameter)
        return cname if cname in available_df['chemical'].values else None


def build_both_out_states():
    both_out_states = []
    for pool_vol, avail_df in grouped_by_vol.items():
        starts = avail_df[['ph','alkalinity','chlorine','calcium_hardness','cyanuric_acid']].drop_duplicates()
        for _, start in starts.iterrows():
            ph_val, alk_val, chlorine, calcium, cya = start
            alk_target, ph_target = 80, 7.4
            alk_chem = pick_chemical('alkalinity', alk_val, alk_target, avail_df)
            if not alk_chem: continue
            rowA = avail_df[(avail_df['ph'] == ph_val) & (avail_df['alkalinity'] == alk_val) &
                            (avail_df['chlorine'] == chlorine) & (avail_df['calcium_hardness'] == calcium) &
                            (avail_df['cyanuric_acid'] == cya) & (avail_df['parameter'] == 'alkalinity') &
                            (avail_df['chemical'] == alk_chem) & (avail_df['target'] == alk_target)]
            if rowA.empty: continue
            ph_chem = pick_chemical('ph', ph_val, ph_target, avail_df)
            if not ph_chem: continue
            rowP = avail_df[(avail_df['ph'] == ph_val) & (avail_df['alkalinity'] == alk_val) &
                            (avail_df['chlorine'] == chlorine) & (avail_df['calcium_hardness'] == calcium) &
                            (avail_df['cyanuric_acid'] == cya) & (avail_df['parameter'] == 'ph') &
                            (avail_df['chemical'] == ph_chem) & (avail_df['target'] == ph_target)]
            if rowP.empty: continue
            both_out_states.append((pool_vol, rowA.iloc[0].to_list(), rowP.iloc[0].to_list()))
    return both_out_states

BOTH_OUT_STATES = build_both_out_states()
print(f"✅ Precomputed {len(BOTH_OUT_STATES)} both_out states")


def perturb_readings(row_list):
    row_list = row_list.copy()
    row_list[0] = rand_weighted('ph')
    row_list[1] = rand_weighted('alkalinity')
    row_list[2] = rand_weighted('chlorine')
    row_list[3] = rand_weighted('calcium_hardness')
    row_list[4] = rand_weighted('cyanuric_acid')
    return row_list


def generate_rows(n_rows=1000, both_out_prob=BOTH_OUT_PROB):
    results = []
    while len(results) < n_rows:
        if random.random() < both_out_prob and BOTH_OUT_STATES:
            _, rowA, rowP = random.choice(BOTH_OUT_STATES)
            for _ in range(ALK_PRIORITY_MULTIPLIER):
                results.append(perturb_readings(rowA)) 
            results.append(perturb_readings(rowP))     
        else:
            results.append(perturb_readings(df_base.sample(1).iloc[0].to_list()))
    return pd.DataFrame(results, columns=df_base.columns)


if __name__ == "__main__":
    df_out = generate_rows(50000)
    df_out.to_csv("synthetic_high_variance_weighted_extremes.csv", index=False)
    print(f"✅ Generated {len(df_out)} rows → synthetic_high_variance_weighted_extremes.csv")
