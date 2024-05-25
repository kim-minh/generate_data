from itertools import combinations_with_replacement, permutations
import numpy as np
import pandas as pd

NUM_SAMPLES = 1000

rng = np.random.default_rng()
mean = rng.integers(1, 6, size=[NUM_SAMPLES, 10])

df1 = pd.DataFrame(index=range(0, NUM_SAMPLES),
                   columns=['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4', 'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8'])
df2 = pd.DataFrame(index=range(0, NUM_SAMPLES), columns=['BL 1', 'BL 2', 'BL 3', 'BL 4', 'BL 5'])

df = df1.join(
    pd.DataFrame(mean, columns=['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6', 'SI 1', 'SI 2', 'SI 3', 'SI 4']))
df = df.join(df2)

df["Mean_CSR"] = pd.Series(dtype='float64')
df["Mean_PI"] = df[['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6']].mean(axis=1)
df["Mean_SI"] = df[['SI 1', 'SI 2', 'SI 3', 'SI 4']].mean(axis=1)


def find_combinations(num, target):
    data = [1, 2, 3, 4, 5]
    target = target * num

    combination_arr = [combination for combination in combinations_with_replacement(data, num) if
                       sum(combination) == target]
    combinations_index = rng.integers(0, len(combination_arr))

    permutation_arr = list(permutations(combination_arr[combinations_index]))
    permutation_index = rng.integers(0, len(permutation_arr))
    return np.array(permutation_arr[permutation_index])


for row in df.itertuples():
    index = getattr(row, 'Index')

    mean_pi = df.at[index, 'Mean_PI']
    mean_si = df.loc[index, 'Mean_SI']

    range_array = np.arange(1, min(mean_pi, mean_si))
    if range_array.size == 0:
        df = df.drop(index=index)
        continue

    mean_csr = rng.choice(np.arange(1, min(mean_pi, mean_si), 0.125))
    if mean_csr < 1.3:
        df = df.drop(index=index)
        continue

    df.loc[index, 'Mean_CSR'] = mean_csr
    df.loc[index, ['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4', 'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8']] = find_combinations(8,
                                                                                                                mean_csr)

array = df["Mean_PI"]
df["Mean_BL"] = np.around(array * 1.2, 1)
df = df.drop(df[df['Mean_BL'] > 5].index)

for index, row in df.iterrows():
    df.loc[index, ['BL 1', 'BL 2', 'BL 3', 'BL 4', 'BL 5']] = find_combinations(5, df.loc[index, 'Mean_BL'])

while not df.empty:
    for row in df.itertuples():
        index = getattr(row, 'Index')

        df.loc[index, 'Center_CSR'] = df.loc[index, 'Mean_CSR'] - np.mean(df['Mean_CSR'])
        df.loc[index, 'Center_PI'] = df.loc[index, 'Mean_PI'] - np.mean(df['Mean_PI'])
        df.loc[index, 'Center_SI'] = df.loc[index, 'Mean_SI'] - np.mean(df['Mean_SI'])
        df.loc[index, 'Center_BL'] = df.loc[index, 'Mean_BL'] - np.mean(df['Mean_BL'])

    df['Interaction_CSR_PI'] = df['Center_CSR'] * df['Center_PI']
    df['Interaction_CSR_SI'] = df['Center_CSR'] * df['Center_SI']

    if (df['Interaction_CSR_PI'] < 0).any() | (df['Interaction_CSR_SI'] < 0).any() | (df['Center_BL'] < -0.8).any():
        df = df.drop(
            df[(df['Interaction_CSR_PI'] < 0) | (df['Interaction_CSR_SI'] < 0) | (df['Center_BL'] < -0.8)].index)
    else:
        break

df.to_excel("data.xlsx", index=False)

print(df)
