#!/usr/bin/env python
# -*- coding: utf-8
"""
Script 2: Proliferation vs Morphology (& DNA Damage)

For each .czi in input_folder:
 1. Read channels, max-project.
 2. Segment on BF.
 3. For each cell:
    - measure proliferation = mean orange within mask,
    - measure morphology: area, perimeter, eccentricity, solidity,
    - count DNA damage foci in red (same pipeline as Script 1).
 4. Save CSV of all metrics.
 5. Plot proliferation vs area, colored by foci_count.
"""
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from czifile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import seaborn as sns

# --- USER PARAMETERS ---
input_folder   = r"J:/Cells/ProliferationStudy"
pattern        = "*.czi"
min_cell_area  = 50
min_foci_area  = 2
# -----------------------

days = ["Day1", "Day2", "Day3"]
fbs_conditions = ["0p5FBS", "10FBS"]

model = StarDist2D.from_pretrained('2D_demo')
records = []

for day in days:
    for fbs in fbs_conditions:
        folder = os.path.join(input_folder, day, fbs)
        for fp in sorted(glob(os.path.join(folder, pattern))):
            base = os.path.splitext(os.path.basename(fp))[0]
            img5d = imread(fp)               # (S,T,Z,Y,X,C)
            print("raw CZI shape:", img5d.shape, "dtype:", img5d.dtype)
            # Reshape to get channels and spatial dimensions
            # img5d shape: (1, 1, 3, 1, 20, 512, 512, 1)
            # Extract (C, Z, Y, X)
            img = img5d[0,0,:,0,:, :, :, 0]  # (3, 20, 512, 512)
            # Max project along Z for each channel
            red    = img[1].max(axis=0)    # (512, 512)
            orange = img[0].max(axis=0)
            bf     = img[2].max(axis=0)

            # Apply Gaussian smoothing before normalization
            sigma = 3
            red_smooth    = gaussian(red,    sigma=1, preserve_range=True)
            orange_smooth = gaussian(orange, sigma=1, preserve_range=True)
            bf_smooth     = gaussian(orange,     sigma=sigma, preserve_range=True)

            # normalize to fixed range (0, 65535) for uint16 images
            red_n    = normalize(red_smooth,    0, 98, axis=None)
            orange_n = orange_smooth
            bf_n     = normalize(bf_smooth,     0, 98, axis=None)

            # segment
            labels,_ = model.predict_instances(bf_n)

            # --- Save overlay image: segmentation on BF channel ---
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ax1.imshow(bf_n, cmap='gray')
            ax1.imshow(labels, cmap='jet', alpha=0.3)
            ax1.set_title('Segmentation on BF')
            ax1.axis('off')
            plt.tight_layout()
            overlay_bf_path = os.path.join(folder, f"{base}_overlay_bf.png")
            fig1.savefig(overlay_bf_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"  → Saved BF overlay: {overlay_bf_path}")

            # --- Save overlay image: segmentation and foci on red channel ---
            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.imshow(red_n, cmap='gray')
            ax2.imshow(labels, cmap='jet', alpha=0.2)
            for region in regionprops(labels, intensity_image=red_n):
                if region.area < min_cell_area:
                    continue
                minr, minc, maxr, maxc = region.bbox
                cell_red = red_n[minr:maxr, minc:maxc]
                mask = region.image
                bg_val = np.median(cell_red[mask])
                corr = cell_red - bg_val
                corr[corr < 0] = 0
                if corr.max() > 0:
                    thr = threshold_otsu(corr)
                    fmask = corr > thr
                else:
                    fmask = np.zeros_like(corr, bool)
                fmask = remove_small_objects(fmask, min_size=min_foci_area)
                for fp in regionprops(label(fmask)):
                    y0, x0 = fp.centroid
                    ax2.plot(minc + x0, minr + y0, 'ro', markersize=4)
            ax2.set_title('Foci on Red')
            ax2.axis('off')
            plt.tight_layout()
            overlay_red_path = os.path.join(folder, f"{base}_overlay_red.png")
            fig2.savefig(overlay_red_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"  → Saved red overlay: {overlay_red_path}")

            # per-cell metrics
            for region in regionprops(labels, intensity_image=orange_n):
                if region.area < min_cell_area:
                    continue
                cid = region.label
                minr, minc, maxr, maxc = region.bbox

                # proliferation intensity
                cell_orange = orange_n[minr:maxr, minc:maxc]
                mask        = region.image
                prolif_int  = cell_orange[mask].mean()

                # morphology
                area      = region.area
                perim     = region.perimeter
                ecc       = region.eccentricity
                solidity  = region.solidity

                # DNA damage foci (red)
                cell_red = red_n[minr:maxr, minc:maxc]
                red_mask = np.zeros_like(cell_red)
                red_mask[mask] = cell_red[mask]
                bg_val = np.median(red_mask[mask])
                corr   = red_mask - bg_val
                corr[corr < 0] = 0
                if corr.max() > 0:
                    thr   = threshold_otsu(corr)
                    fmask = corr > thr
                else:
                    fmask = np.zeros_like(corr, bool)
                fmask = remove_small_objects(fmask, min_size=min_foci_area)
                fcount = len(regionprops(label(fmask)))

                records.append({
                    'file': base,
                    'day': day,
                    'fbs': fbs,
                    'cell_id': cid,
                    'area': area,
                    'perimeter': perim,
                    'eccentricity': ecc,
                    'solidity': solidity,
                    'prolif_intensity': prolif_int,
                    'foci_count': fcount
                })

# save CSV
df2 = pd.DataFrame(records)
csv2 = os.path.join(input_folder, "prolif_morphology_damage.csv")
df2.to_csv(csv2, index=False)
print(f"Saved to {csv2}")

# scatter: proliferation vs area colored by foci
plt.figure(figsize=(6,5))
sc = plt.scatter(df2['area'], df2['prolif_intensity'],
                 c=df2['foci_count'], cmap='viridis',
                 s=20, alpha=0.7)
plt.xlabel('Cell area (px)')
plt.ylabel('Proliferation intensity')
cbar = plt.colorbar(sc)
cbar.set_label('Foci count')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_vs_area_foci.png'), dpi=300)
plt.close()

# Add comparison plots for proliferation and DNA damage by day and FBS
plt.figure(figsize=(7,5))
sns.boxplot(data=df2, x='day', y='prolif_intensity', hue='fbs')
plt.title('Proliferation Intensity by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_by_day_fbs.png'), dpi=300)
plt.close()

plt.figure(figsize=(7,5))
sns.boxplot(data=df2, x='day', y='foci_count', hue='fbs')
plt.title('DNA Damage (Foci Count) by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_by_day_fbs.png'), dpi=300)
plt.close()

# Violin plot: Proliferation intensity by day and FBS
plt.figure(figsize=(8,5))
sns.violinplot(data=df2, x='day', y='prolif_intensity', hue='fbs', split=True, inner='quartile')
plt.title('Proliferation Intensity Distribution by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_violin_by_day_fbs.png'), dpi=300)
plt.close()

# Violin plot: DNA damage (foci count) by day and FBS
plt.figure(figsize=(8,5))
sns.violinplot(data=df2, x='day', y='foci_count', hue='fbs', split=True, inner='quartile')
plt.title('DNA Damage (Foci Count) Distribution by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_violin_by_day_fbs.png'), dpi=300)
plt.close()
