from itertools import combinations_with_replacement, permutations
import numpy as np
import pandas as pd

NUM_SAMPLES = 1000

rng = np.random.default_rng()
mean = rng.integers(1, 6, size=[NUM_SAMPLES, 15])

df = pd.DataFrame(index=range(0, NUM_SAMPLES),
                  columns=['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4', 'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8'])

df = df.join(pd.DataFrame(mean, columns=['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6', 'SI 1', 'SI 2', 'SI 3', 'SI 4',
                                         'BL 1', 'BL 2', 'BL 3', 'BL 4', 'BL 5']))

df["Mean_CSR"] = pd.Series(dtype='float64')
df["Mean_PI"] = df[['PI 1', 'PI 2', 'PI 3', 'PI 4', 'PI 5', 'PI 6']].mean(axis=1)
df["Mean_SI"] = df[['SI 1', 'SI 2', 'SI 3', 'SI 4']].mean(axis=1)


def find_combinations(target):
    data = [1, 2, 3, 4, 5]
    num = 8
    target = target * num

    combination_arr = [combination for combination in combinations_with_replacement(data, num) if
                       sum(combination) == target]
    combinations_index = rng.integers(0, len(combination_arr))

    permutation_arr = list(permutations(combination_arr[combinations_index]))
    permutation_index = rng.integers(0, len(permutation_arr))
    return np.array(permutation_arr[permutation_index])


for index, row in df.iterrows():
    mean_pi = df.loc[index, 'Mean_PI']
    mean_si = df.loc[index, 'Mean_SI']

    range_array = np.arange(1, min(mean_pi, mean_si))
    if range_array.size == 0:
        df = df.drop(index=index)
        continue

    mean_csr = rng.choice(np.arange(1, min(mean_pi, mean_si), 0.125))
    df.loc[index, 'Mean_CSR'] = mean_csr
    df.loc[index, ['CSR 1', 'CSR 2', 'CSR 3', 'CSR 4', 'CSR 5', 'CSR 6', 'CSR 7', 'CSR 8']] = find_combinations(mean_csr)


array = df["Mean_PI"] / df["Mean_SI"]
df["Mean_BL"] = np.around(array * 3, 1)

while not df.empty:
    for index, row in df.iterrows():
        df.loc[index, 'Center_CSR'] = df.loc[index, 'Mean_CSR'] - np.mean(df['Mean_CSR'])
        df.loc[index, 'Center_PI'] = df.loc[index, 'Mean_PI'] - np.mean(df['Mean_PI'])
        df.loc[index, 'Center_SI'] = df.loc[index, 'Mean_SI'] - np.mean(df['Mean_SI'])
        df.loc[index, 'Center_BL'] = df.loc[index, 'Mean_BL'] - np.mean(df['Mean_BL'])

    df['Interaction_CSR_PI'] = df['Center_CSR'] * df['Center_PI']
    df['Interaction_CSR_SI'] = df['Center_CSR'] * df['Center_SI']

    if (df['Interaction_CSR_PI'] < 0).any() | (df['Interaction_CSR_SI'] < 0).any() | (df['Center_BL'] < -0.2).any():
        df = df.drop(df[(df['Interaction_CSR_PI'] < 0) | (df['Interaction_CSR_SI'] < 0) | (df['Center_BL'] < -0.2)].index)
    else:
        break

    # if (df['Interaction_CSR_PI'] < 0).any() | (df['Interaction_CSR_SI'] < 0).any():
    #     df = df.drop(df[(df['Interaction_CSR_PI'] < 0) | (df['Interaction_CSR_SI'] < 0)].index)
    # else:
    #     break

df.reset_index(drop=True)
df.to_excel("data.xlsx")

print(df)
