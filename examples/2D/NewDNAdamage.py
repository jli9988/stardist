#!/usr/bin/env python
# -*- coding: utf-8
"""
Dose & Time vs DNA Damage (red channel) using yellow channel for segmentation.

Folder structure:
  input_folder/
    Day2/
      A...czi  # 1 μM
      B...czi  # 5 μM
      ...
    Day3/
    Day5/
"""

import os, re, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from czifile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# — USER PARAMETERS —
input_folder  = r"J:/Cells/OlaparibDosing"
days          = ["Day2","Day3","Day5"]
min_cell_area = 50     # px
min_foci_area = 2      # px
# — End parameters —

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0; change if needed
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {[d.name for d in physical_devices]}")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")
else:
    print("No GPU found, running on CPU.")

# StarDist model
model = StarDist2D.from_pretrained('2D_demo')

# mapping filename prefix → dose
dose_map = {'A':1, 'B':5, 'C':10, 'D':20, 'E':0}

records = []
for day in days:
    time_pt = int(day.replace("Day",""))  # 2, 3, or 5
    folder  = os.path.join(input_folder, day)
    for fp in sorted(glob(os.path.join(folder, "*.czi"))):
        fname = os.path.basename(fp)
        base  = os.path.splitext(fname)[0]
        prefix = base[0].upper()
        dose   = dose_map.get(prefix, np.nan)

        #--- load and max‐project ---
        img5d = imread(fp)            # (S,T,Z,Y,X,C)
        img = img5d[0,0,:,0,0,:,:,0]    # Extract channels and spatial dimensions (C,H,W)
        red    = img[2]    # First channel
        green  = img[0]    # Fourth channel
        yellow = img[1]    # Second channel
        bf     = img[3]    # Third channel

        # Apply Gaussian smoothing before normalization
        sigma = 2
        yellow_smooth = gaussian(yellow, sigma=sigma, preserve_range=True)
        red_smooth    = gaussian(red,    sigma=0.5, preserve_range=True)

        # normalize for segmentation & measurement
        yellow_n = normalize(yellow_smooth, axis=(0,1))
        red_n    = normalize(red_smooth,    axis=(0,1))

        # segment cells on yellow channel
        labels, _ = model.predict_instances(yellow_n)

        # per‐cell foci counting
        for region in regionprops(labels, intensity_image=red_n):
            if region.area < min_cell_area:
                continue
            cid = region.label

            # mask + background subtraction
            mask      = region.image               # bool mask
            cell_red  = region.intensity_image     # red_n in bbox
            bg_val    = np.median(cell_red[mask])
            corr      = cell_red - bg_val
            corr[corr < 0] = 0

            # Otsu + prune
            if corr.max() > 0:
                thr   = threshold_otsu(corr)
                fmask = corr > thr
            else:
                fmask = np.zeros_like(corr, bool)
            fmask = remove_small_objects(fmask, min_size=min_foci_area)

            # count foci
            fcount = len(regionprops(label(fmask)))

            # proliferation intensity (mean yellow within mask)
            cell_yellow = region.intensity_image if region.intensity_image.shape == region.image.shape else yellow_n[region.slice][region.image]
            prolif_int = cell_yellow[region.image].mean()

            records.append({
                'file': base,
                'dose': dose,
                'time': time_pt,
                'cell_id': cid,
                'foci_count': fcount,
                'prolif_intensity': prolif_int
            })

        # --- Save overlay image: segmentation on yellow channel ---
        fig1, ax1 = plt.subplots(figsize=(6,6))
        ax1.imshow(yellow_n, cmap='gray')
        ax1.imshow(labels, cmap='jet', alpha=0.3)
        ax1.set_title('Segmentation on Yellow')
        ax1.axis('off')
        plt.tight_layout()
        overlay_yellow_path = os.path.join(folder, f"{base}_overlay_yellow.png")
        fig1.savefig(overlay_yellow_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"  → Saved yellow overlay: {overlay_yellow_path}")

        # --- Save overlay image: segmentation and foci on red channel ---
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.imshow(red_n, cmap='gray')
        ax2.imshow(labels, cmap='jet', alpha=0.2)
        # Overlay foci centroids
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

# save combined CSV
df = pd.DataFrame(records)
out_csv = os.path.join(input_folder, "dna_damage_all.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# — Plot average foci vs dose for each time —
plt.figure(figsize=(6,4))
for t in sorted(df['time'].unique()):
    sub = df[df['time']==t]
    means = sub.groupby('dose')['foci_count'].mean()
    plt.plot(means.index, means.values, 'o-',
             label=f"{t} days")
plt.xlabel('Dose (µM)')
plt.ylabel('Avg foci per cell')
plt.legend(title='Time')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'avg_foci_vs_dose.png'), dpi=300)
plt.close()

# — Plot PDF of foci counts by dose (all times) —
import seaborn as sns
plt.figure(figsize=(6,4))
sns.kdeplot(data=df, x='foci_count', hue='dose',
            common_norm=False, fill=False)
# plt.yscale('log')
plt.xlabel('Foci per cell')
plt.ylabel('Density (log scale)')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_pdf_by_dose.png'), dpi=300)
plt.close()

# 1) Average foci vs time for each dose
plt.figure(figsize=(6,4))
for d in sorted(df['dose'].unique()):
    sub   = df[df['dose']==d]
    meanf = sub.groupby('time')['foci_count'].mean()
    plt.plot(meanf.index, meanf.values, '-o', label=f'{d} μM')
plt.xlabel('Time (days)')
plt.ylabel('Avg foci per cell')
plt.title('Average DNA damage vs Time')
plt.legend(title='Dose')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'avg_foci_vs_time.png'), dpi=300)
plt.close()

# 2) Distribution of foci counts by time & dose
plt.figure(figsize=(8,6))
sns.boxplot(
    data=df,
    x='time',
    y='foci_count',
    hue='dose',
    palette='Set2'
)
# plt.yscale('log')
plt.xlabel('Time (days)')
plt.ylabel('Foci count (log scale)')
plt.title('Foci-count distributions by Time & Dose')
plt.legend(title='Dose', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_boxplot_time_dose.png'), dpi=300)
plt.close()

# 3) Average proliferation intensity vs time for each dose
plt.figure(figsize=(6,4))
for d in sorted(df['dose'].unique()):
    sub = df[df['dose']==d]
    meanp = sub.groupby('time')['prolif_intensity'].mean()
    plt.plot(meanp.index, meanp.values, '-o', label=f'{d} μM')
plt.xlabel('Time (days)')
plt.ylabel('Avg proliferation intensity')
plt.title('Average Proliferation Intensity vs Time')
plt.legend(title='Dose')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'avg_prolif_vs_time.png'), dpi=300)
plt.close()

# 4) Boxplot of proliferation intensity by time & dose with error bars
plt.figure(figsize=(8,6))
sns.boxplot(
    data=df,
    x='time',
    y='prolif_intensity',
    hue='dose',
    palette='Set2',
    showfliers=False
)
sns.pointplot(
    data=df,
    x='time',
    y='prolif_intensity',
    hue='dose',
    dodge=0.4,
    join=False,
    palette='dark:k',
    errorbar='sd',
    markers='D',
    scale=0.7,
    errwidth=1.5,
    capsize=0.1
)
plt.xlabel('Time (days)')
plt.ylabel('Proliferation intensity')
plt.title('Proliferation Intensity by Time & Dose')
plt.legend(title='Dose', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_boxplot_time_dose.png'), dpi=300)
plt.close()

# 5) Scatter plot: proliferation intensity vs DNA foci count
plt.figure(figsize=(6,5))
plt.scatter(df['prolif_intensity'], df['foci_count'], alpha=0.6, s=20)
plt.xlabel('Proliferation intensity')
plt.ylabel('DNA foci count')
plt.title('Proliferation Intensity vs DNA Foci Count')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_vs_foci_scatter.png'), dpi=300)
plt.close()
