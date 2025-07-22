import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from czifile import imread
from skimage.filters import threshold_otsu

input_folder = r"J:/Cells/ProliferationStudy"
days = ["Day1", "Day2", "Day3"]
fbs_conditions = ["0p5FBS", "10FBS"]
pattern = "*.czi"

records = []
for day in days:
    for fbs in fbs_conditions:
        folder = os.path.join(input_folder, day, fbs)
        for fp in sorted(glob(os.path.join(folder, pattern))):
            base = os.path.splitext(os.path.basename(fp))[0]
            img5d = imread(fp)
            # img5d shape: (1, 1, 3, 1, 20, 512, 512, 1)
            img = img5d[0,0,:,0,:, :, :, 0]  # (3, 20, 512, 512)
            # Max project along Z for each channel
            red    = img[1].max(axis=0)
            orange = img[0].max(axis=0)
            # Remove noise/background using Otsu threshold
            for channel_name, channel_img in zip(['orange','red'], [orange, red]):
                try:
                    thr = threshold_otsu(channel_img)
                except Exception:
                    thr = np.percentile(channel_img, 10)  # fallback if Otsu fails
                mask = channel_img > thr
                intensities = channel_img[mask]
                for val in intensities:
                    records.append({
                        'day': day,
                        'fbs': fbs,
                        'file': base,
                        'channel': channel_name,
                        'intensity': val
                    })

# Save as CSV for reference
raw_csv = os.path.join(input_folder, 'raw_pixel_intensities.csv')
pd.DataFrame(records).to_csv(raw_csv, index=False)
print(f'Saved raw intensities to {raw_csv}')

# Plot distributions
for channel in ['orange','red']:
    plt.figure(figsize=(10,6))
    sns.violinplot(
        data=pd.DataFrame(records)[pd.DataFrame(records)['channel']==channel],
        x='day', y='intensity', hue='fbs', split=True, inner='quartile', cut=0
    )
    plt.title(f'Pixel Intensity Distribution ({channel}) by Day and FBS')
    plt.tight_layout()
    plt.savefig(os.path.join(input_folder, f'raw_{channel}_violin_by_day_fbs.png'), dpi=300)
    plt.close()

print('All raw intensity plots saved to', input_folder)
