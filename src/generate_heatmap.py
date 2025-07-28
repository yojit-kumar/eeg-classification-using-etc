import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def plot_heatmap(df, result_dir):
    df['delta'] = df['ETC_EyesOpen'] - df['ETC_EyesClosed']
    avg_delta = df.groupby('channel')['delta'].mean()


    electrode_coords = {
        "Fp1.": (4,9), "Fpz.": (5,9), "Fp2.": (6,9),
        "Af7.": (1,8), "Af3.": (3,8), "Afz.": (5,8), "Af4.": (7,8), "Af8.": (9,8), 
        "F7..": (1,7), "F5..": (2,7), "F3..": (3,7), "F1..": (4,7), "Fz..": (5,7), "F2..": (6,7), "F4..": (7,7), "F6..": (8,7), "F8..": (9,7),
        "Ft7.": (1,6), "Fc5.": (2,6), "Fc3.": (3,6), "Fc1.": (4,6), "Fcz.": (5,6), "Fc2.": (6,6), "Fc4.": (7,6), "Fc6.": (8,6), "Ft8.": (9,6),
        "T9..": (0,5), "T7..": (1,5), "C5..": (2,5), "C3..": (3,5), "C1..": (4,5), "Cz..": (5,5), "C2..": (6,5), "C4..": (7,5), "C6..": (8,5), "T8..": (9,5), "T10.": (10,5),
        "Tp7.": (1,4), "Cp5.": (2,4), "Cp3.": (3,4), "Cp1.": (4,4), "Cpz.": (5,4), "Cp2.": (6,4), "Cp4.": (7,4), "Cp6.": (8,4), "Tp8.": (9,4),
        "P7..": (1,3), "P5..": (2,3), "P3..": (3,3), "P1..": (4,3), "Pz..": (5,3), "P2..": (6,3), "P4..": (7,3), "P6..": (8,3), "P8..": (9,3),
        "Po7.": (1,2), "Po3.": (3,2), "Poz.": (5,2), "Po4.": (7,2), "Po8.": (9,2),
        "O1..": (4,1), "Oz..": (5,1), "O2..": (6,1),
        "Iz..": (5,0)
    }

    # Clean channel names
    new_coords = {}
    for ch in list(electrode_coords.keys()):
        v = electrode_coords[ch]
        clean_ch = ch.strip('.')
        if 'Fp' not in clean_ch:
            clean_ch = clean_ch.upper()
        if clean_ch.endswith('Z'):
            clean_ch = clean_ch[:-1] + 'z'
        new_coords[clean_ch] = v

    electrode_coords = new_coords


    plt.rcParams.update({
        "font.family": "serif",
        })

    # Create heatmap grid
    heatmap = np.full((11, 11), np.nan)
    for ch, delta_val in avg_delta.items():
        if ch in electrode_coords:
            x, y = electrode_coords[ch]
            heatmap[y, x] = delta_val


    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = sns.diverging_palette(210, 30, as_cmap=True)
    sns.heatmap(
        heatmap, 
        cmap=cmap, 
        square=True,
        mask=np.isnan(heatmap), 
        ax=ax,
        linewidths=0.5
    )

    # Annotate electrodes
    for ch, (x, y) in electrode_coords.items():
        if heatmap[y, x] is not np.nan:
            ax.text(
                x + 0.5, y + 0.5, ch, 
                ha='center', va='center',
                fontsize=10, color='black'
                )

    # Final adjustments
    ax.set_title("Average ETC Difference across all Channels", fontsize=16)
    ax.axis('off')  # Remove axis ticks and labels
    plt.tight_layout()

    timestr = time.strftime("%y%m%d-%H%M%S")
    result_path = os.path.join(result_dir, f"etc_difference_heatmap_{timestr}.png")

    # Save high-resolution figure
    plt.savefig(result_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    resut_dir = "../results/figures/"
