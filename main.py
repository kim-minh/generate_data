from itertools import combinations_with_replacement, permutations
import numpy as np
import pandas as pd

NUM_SAMPLES = 1000

rng = np.random.default_rng()
values = rng.integers(1, 6, size=[NUM_SAMPLES, 10])

df = pd.DataFrame(values, columns=['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6', 'SI 1', 'SI 2', 'SI 3', 'SI 4'])
df = pd.concat([df, pd.DataFrame(index=range(NUM_SAMPLES),
                                 columns=['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4', 'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8',
                                          'BL 1', 'BL 2', 'BL 3', 'BL 4', 'BL 5'])], axis=1)

df["Mean_PI"] = df[['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6']].mean(axis=1)
df["Mean_SI"] = df[['SI 1', 'SI 2', 'SI 3', 'SI 4']].mean(axis=1)


def find_combinations(num, target):
    data = [1, 2, 3, 4, 5]
    total = target * num
    combination_arr = [comb for comb in combinations_with_replacement(data, num) if sum(comb) == total]
    if not combination_arr:
        return np.array([])
    comb_index = rng.integers(0, len(combination_arr))
    permutation_arr = list(permutations(combination_arr[comb_index]))
    perm_index = rng.integers(0, len(permutation_arr))
    return np.array(permutation_arr[perm_index])


valid_rows = df[(df["Mean_PI"] > 1.3) & (df["Mean_SI"] > 1.3)].index
df = df.loc[valid_rows]

# Generate Mean_CSR based on valid ranges
mean_csr_values = []
for row in df.itertuples():
    mean_pi = getattr(row, 'Mean_PI')
    mean_si = getattr(row, 'Mean_SI')
    valid_range = np.arange(1, min(mean_pi, mean_si), 0.125)
    if valid_range.size > 0:
        mean_csr = rng.choice(valid_range)
        if mean_csr >= 1.3:
            mean_csr_values.append(mean_csr)
        else:
            mean_csr_values.append(np.nan)
    else:
        mean_csr_values.append(np.nan)

df["Mean_CSR"] = mean_csr_values
df = df.dropna(subset=["Mean_CSR"])

array = df["Mean_PI"] / df["Mean_SI"]
df["Mean_BL"] = np.round(array * 3, 1)
df = df[(df["Mean_BL"] * 10) % 2 == 0]
df = df[df["Mean_BL"] <= 5]

while True:
    df['Center_CSR'] = df['Mean_CSR'] - df['Mean_CSR'].mean()
    df['Center_PI'] = df['Mean_PI'] - df['Mean_PI'].mean()
    df['Center_SI'] = df['Mean_SI'] - df['Mean_SI'].mean()
    df['Center_BL'] = df['Mean_BL'] - df['Mean_BL'].mean()

    df['Interaction_CSR_PI'] = df['Center_CSR'] * df['Center_PI']
    df['Interaction_CSR_SI'] = df['Center_CSR'] * df['Center_SI']

    conditions = (df['Interaction_CSR_PI'] < 0) | (df['Interaction_CSR_SI'] < 0) | (df['Center_BL'] < -0.5)
    if not conditions.any():
        break
    df = df[~conditions]

for row in df.itertuples():
    index = getattr(row, 'Index')
    mean_csr = getattr(row, 'Mean_CSR')
    mean_bl = getattr(row, 'Mean_BL')

    df.loc[index, ['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4',
                   'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8']] = find_combinations(8, mean_csr)
    df.loc[index, ['BL 1', 'BL 2', 'BL 3', 'BL 4', 'BL 5']] = find_combinations(5, mean_bl)

df.to_excel("data.xlsx", index=False)
print(df)
