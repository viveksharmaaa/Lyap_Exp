import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D



plt.rcParams.update({
    "font.size": 16,           # base font size
    "axes.titlesize": 18,      # title size
    "axes.labelsize": 16,      # x/y label size
    "xtick.labelsize": 14,     # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# Change the data_path to point to the .npz file on your computer.
data_path =    "/home/sharma/Projects/DDAT/Lyap_Exp/results_random_weight_50seed.npz" #"/home/sharma/Projects/DDAT/Lyap_Exp/results_random_lr_20seed.npz" #
data = np.load(data_path) #lr with random weights

# Extract arrays
lams      = data["lambda_max"]        # HVP λ_max
lams_dns  = data["lambda_max_dns"]    # DNS λ_max
errs      = data["test_loss"]         # test loss
trains    = data["train_loss"]        # train loss
gen_gaps  = data["gen_gap"]           # generalization gap
widths    = data["width"]             # width of hidden layer
lrs       = data["lr"]                # learning rate
seeds     = data["seed"]              # seed
sigma_max = data["sigma_max"] if "sigma_max" in data.files else None       # sigma_max
print("Loaded:", len(lams), "entries")

if "_lr_" in data_path:
    tag = "lr"
elif "weight" in data_path:
    tag = "weight"
else:
    tag = "generic"

# ============================================================
# 1.  Correlation analysis between test_loss and HVP λ_max
# ============================================================
pearson_r, p_val = pearsonr(lams, errs)
rho, p_rho = spearmanr(lams, errs)

print("\n========== GLOBAL CORRELATION RESULTS ==========")
print("Pearson r:", pearson_r, "p-value:", p_val)
print("Spearman ρ:", rho, "p-value:", p_rho)

if p_val < 0.05:
    print("✔️ Statistically significant (p < 0.05)")
else:
    print("⚠️ Not statistically significant (p ≥ 0.05)")

# --------------------------------------------------------
# 2. Density plot with scatter points for different lr
# --------------------------------------------------------

if "_lr_" in data_path:
    print("This dataset varies learning rate.")
    marker_map = {
        1e-5: "o",  # circle
        1e-4: "s",  # square
        1e-3: "D",  # diamond
        1e-2: "^",  # upward triangle
        5e-2: "v",  # downward triangle
    }
    color_map = {
        1e-5: "blue",
        1e-4: "green",
        1e-3: "orange",
        1e-2: "red",
        5e-2: "yellow"
    }
elif "weight" in data_path:
    print("This dataset uses random parameter initializations")
    marker_map={1e-2: "s"}
    color_map={1e-2: "blue"}


# Scaler
# absolute max value (ignore sign)
max_abs = np.max(np.abs(lams))

# compute exponent (power of 10)
exp = int(np.floor(np.log10(max_abs)))

# compute the scale to multiply data by
scale = 10**(-exp)

# --------------------------------------------------------
# 3. Minimum Viable Product Plots
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# -------- Panel A : Distribution of Test Loss vs. Max LE  -------- #
ax = axes[0]
sns.kdeplot(
    x=scale*lams,
    y=errs,
    fill=True,
    cmap="mako",
    levels=30,
    ax=ax
)

unique_lrs = np.unique(lrs)

if "_lr_" in data_path:
    palette = sns.color_palette("colorblind", n_colors=len(unique_lrs))
    sns.scatterplot(
        x=scale * lams,
        y=errs,
        hue=lrs,
        palette=palette,  # or "tab10", "coolwarm", etc.
        s=40,
        edgecolor="black",
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    # Convert raw lr labels to nicer LaTeX labels
    new_labels = [fr"$\eta = {float(l):.0e}$" for l in labels[:]]  # skip "hue" label
    ax.legend(
        handles=handles[:],  # drop seaborn's "hue" label
        labels=new_labels,
        fontsize=12,
        title_fontsize=13,
        frameon=False
    )
elif "weight" in data_path:
    # Custom legend entry (manual)
    lr_val = unique_lrs[0]


    sns.scatterplot(
        x=scale*lams,
        y=errs,
        s=25,
        color="green",
        edgecolor="black",
        ax=ax
    )

    custom_handle = Line2D(
        [0], [0],
        marker="o",
        color="tab:green",
        linestyle="",
        markersize=8,
        label=fr"$\eta = {lr_val:.0e}$"
    )
    ax.legend(handles=[custom_handle], frameon=False, fontsize=12)

ax.set_title("(a): Distribution of Test Loss vs. $\lambda_{max}$")
ax.set_xlabel(r"$\lambda_{\max}$")
ax.set_ylabel("test_loss")
# ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
# ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xlim(right=3) #remove outlier
ax.ticklabel_format(style='plain', axis='x')          # disable scientific notation
ax.xaxis.get_offset_text().set_visible(False)   # hide the '1e-4' offset text
#ax.set_xlabel(r"$\lambda_{\max} \;(\times 10^{-4})$")
ax.set_xlabel(rf"$\lambda_{{\max}} \;(\times 10^{{{exp}}})$")


# -------- Panel B : Correlation Matrix for Test Loss vs. Max LE  -------- #
ax = axes[1]
corr = np.corrcoef(lams, errs)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    xticklabels=[r"$\lambda_{max}$", "test_loss"],
    yticklabels=[r"$\lambda_{max}$", "test_loss"],
    ax=ax
)
ax.set_title("(b) Correlation Matrix")
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
outfile = f"Minimum_Viable_product_two_panel_{tag}.png"
plt.savefig(outfile, dpi=300)
plt.show()

if "_lr_" in data_path:

    #Plots for (a) test_loss vs learning rate  (b) lambda_max vs learning rate and (c)
    # Unique LR values
    unique_lrs = np.unique(lrs)

    # Mean and std for each metric grouped by learning rate
    mean_test = [errs[lrs == lr].mean() for lr in unique_lrs]
    std_test = [errs[lrs == lr].std() for lr in unique_lrs]

    mean_lam = [lams[lrs == lr].mean() for lr in unique_lrs]
    std_lam = [lams[lrs == lr].std() for lr in unique_lrs]

    mean_sigma = [sigma_max[lrs == lr].mean() for lr in unique_lrs]
    std_sigma = [sigma_max[lrs == lr].std() for lr in unique_lrs]

    # ---------------- FIGURE ----------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # ========== (a) Test Loss ==========
    ax[0].errorbar(unique_lrs, mean_test, yerr=std_test,
                   marker='o', color='red', capsize=4, linewidth=2)
    ax[0].set_xscale('log')
    ax[0].set_title("(a) Test Loss vs Learning Rate", fontsize=14)
    ax[0].set_xlabel(r"Learning rate ($\eta$)")
    ax[0].set_ylabel("test_loss")
    ax[0].grid(alpha=0.3)

    # ========== (b) λ_max ==========
    ax[1].errorbar(unique_lrs, mean_lam, yerr=std_lam,
                   marker='o', color='blue', capsize=4, linewidth=2)
    ax[1].set_xscale('log')
    #ax[1].set_yscale('log')         # IMPORTANT: λ_max usually spans orders of magnitude
    ax[1].set_title(r"(b) Largest Lyapunov Exponent ($\lambda_{\max}$)", fontsize=14)
    ax[1].set_xlabel(r"Learning rate ($\eta$)")
    ax[1].set_ylabel(r"$\lambda_{\max}$")
    ax[1].grid(alpha=0.3)


    # ========== (c) σ_max ==========
    # OPTIONAL: add theoretical reference line (e.g. 2/η)
    ax[2].plot(unique_lrs, 2/unique_lrs, 'k--', label=r"$2/\eta$")
    ax[2].errorbar(unique_lrs, mean_sigma, yerr=std_sigma,
                   marker='o', color='green', capsize=4, linewidth=2)
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')   # Hessian spectral norm often grows ~1/η^2 or diverges
    ax[2].set_title(r"(c) Maximum Hessian eigenvalue ($\sigma_{\max}$)", fontsize=14)
    ax[2].set_xlabel(r"Learning rate ($\eta$)")
    ax[2].set_ylabel(r"$\sigma_{\max}$")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    outfile = f"three_panel_{tag}.png"
    plt.savefig(outfile, dpi=300)
    plt.show()




