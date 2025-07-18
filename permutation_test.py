# ----------------- path -----------------
high_path = r"D:\Projects\Desicion Mkaing\Data\ds002878-download\Data_mat\100ROI\High_tH_z.npy"
low_path  = r"D:\Projects\Desicion Mkaing\Data\ds002878-download\Data_mat\100ROI\Low_tH_z.npy"

out_pref  = r"D:\Projects\Desicion Mkaing\Data\ds002878-download\Data_mat\100ROI\perm_meanDif\New folder\perm_meanDiff"  
n_perm = 500
alpha  = 0.05
# -------------------------------------------------

import numpy as np
from tqdm import trange

High = np.load(high_path)   # (nHigh, 100, 100)
Low  = np.load(low_path)    # (nLow , 100, 100)
nHigh, nROI, _ = High.shape
nLow            = Low.shape[0]

tri = np.triu_indices(nROI, 1)

# 1) تفاوت میانگین مشاهده‌ای
mean_H = High.mean(axis=0)
mean_L = Low.mean(axis=0)
diff_obs = mean_H - mean_L          # 100×100
abs_diff_obs = np.abs(diff_obs)

# 2) (max |Δmean|)
all_data = np.concatenate([High, Low], axis=0)
null_max = np.zeros(n_perm)

for p in trange(n_perm, desc='Permuting'):
    np.random.shuffle(all_data)
    groupA = all_data[:nHigh]
    groupB = all_data[nHigh:]
    diff_perm = groupA.mean(axis=0) - groupB.mean(axis=0)
    null_max[p] = np.max(np.abs(diff_perm[tri]))

# 3) p-value 
p_corr = np.ones((nROI, nROI))
for i, j in zip(*tri):
    p_val = np.mean(null_max >= abs_diff_obs[i, j])
    p_corr[i, j] = p_corr[j, i] = p_val

sigMask = p_corr < alpha
print(f'#Edges surviving max-stat on |Δmean|: {sigMask.sum()//2}')

# 4) save
np.save(out_pref + '_diff_obs.npy', diff_obs)
np.save(out_pref + '_p_corr.npy',   p_corr)
np.save(out_pref + '_sigMask.npy',  sigMask)
np.save(out_pref + '_null_max.npy', null_max)
print('Saved outputs with prefix', out_pref)
