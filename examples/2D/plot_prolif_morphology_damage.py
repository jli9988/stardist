import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set input folder and CSV path
input_folder = r"J:/Cells/ProliferationStudy"
csv_path = os.path.join(input_folder, "prolif_morphology_damage.csv")

# Load data
print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path)

# Proliferation vs area colored by foci count
plt.figure(figsize=(6,5))
sc = plt.scatter(df['area'], df['prolif_intensity'],
                 c=df['foci_count'], cmap='viridis',
                 s=20, alpha=0.7)
plt.xlabel('Cell area (px)')
plt.ylabel('Proliferation intensity')
cbar = plt.colorbar(sc)
cbar.set_label('Foci count')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_vs_area_foci.png'), dpi=300)
plt.close()

# Boxplot: Proliferation intensity by day and FBS
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='day', y='prolif_intensity', hue='fbs')
plt.title('Proliferation Intensity by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_by_day_fbs.png'), dpi=300)
plt.close()

# Boxplot: DNA damage (foci count) by day and FBS
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='day', y='foci_count', hue='fbs')
plt.title('DNA Damage (Foci Count) by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_by_day_fbs.png'), dpi=300)
plt.close()

# Violin plot: Proliferation intensity by day and FBS
plt.figure(figsize=(8,5))
sns.violinplot(data=df, x='day', y='prolif_intensity', hue='fbs', split=True, inner='quartile')
plt.title('Proliferation Intensity Distribution by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'prolif_violin_by_day_fbs.png'), dpi=300)
plt.close()

# Violin plot: DNA damage (foci count) by day and FBS
plt.figure(figsize=(8,5))
sns.violinplot(data=df, x='day', y='foci_count', hue='fbs', split=True, inner='quartile')
plt.title('DNA Damage (Foci Count) Distribution by Day and FBS')
plt.tight_layout()
plt.savefig(os.path.join(input_folder, 'foci_violin_by_day_fbs.png'), dpi=300)
plt.close()

print('All plots saved to', input_folder)
