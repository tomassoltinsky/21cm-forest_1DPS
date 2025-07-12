import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob

for filename in glob.glob("saved_output/inference_*h/*/test_results.csv"):
    if "archive" in filename:
        continue
    path = filename.split("/")
    telescope = path[1].split('_')[1]
    feature = path[2].split('_')[0]
    if feature not in ['noisy', 'denoised', 'latent', 'laten1']:
        continue
    print(f'processing telescope={telescope}, feature={feature}')
    # ------- 1. Read the data -------------------------------------------------------
    # The CSV file lives on the notebook file‑system at this path
    df = pd.read_csv(filename)

    # The file has four columns:  (x, y) sample coordinates +
    # two columns holding a *test* (x, y) value that identifies which
    # of the five samples each row belongs to.
    df.columns = ["x", "y", "test_x", "test_y"]

    # ------- 2. Prepare the figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    # A common evaluation grid for all KDEs
    x_min, x_max = df["x"].min() - 0.1, df["x"].max() + 0.1
    y_min, y_max = df["y"].min() - 0.1, df["y"].max() + 0.1
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_positions = np.vstack([XX.ravel(), YY.ravel()])

    # Containers for the density values at the five test points
    kde_at_test = []
    product = 1.0
    avg = 0.0

    # ------- 3. Loop over the five samples -----------------------------------------
    unique_tests = df[["test_x", "test_y"]].drop_duplicates().reset_index(drop=True)

    for idx, row in unique_tests.iterrows():
        tx, ty = row["test_x"], row["test_y"]

        # Extract rows that share this test point ⇒ one sample
        mask = (df["test_x"] == tx) & (df["test_y"] == ty)
        sample = df.loc[mask, ["x", "y"]].values.T        # shape (2, N)
        xs, ys = sample

        # Scatter plot of this sample (colour comes from Matplotlib’s default cycle)
        ax.scatter(xs, ys, s=5, alpha=0.3, label=f"Sample {idx + 1}", c='grey')

        # Fit & evaluate the Gaussian KDE
        kde = gaussian_kde(sample)
        ZZ = kde(grid_positions).reshape(XX.shape)

        dx = (x_max - x_min) / 199           # 200 grid points → 199 intervals
        dy = (y_max - y_min) / 199
        total_prob = (ZZ.sum() * dx * dy)
        print(total_prob)                    # 0.999…  (very close to 1)

        # Overlay KDE contours
        contours = ax.contour(XX, YY, ZZ, levels=30, linewidths=1.0, alpha=0.8,  cmap='plasma')

        # Plot the test value as a star
        ax.plot(tx, ty, marker="*", markersize=12, color="black")

        # Store the KDE value at its own test point
        dx_tile = 0.05
        dy_tile = 0.2 
        val = kde([tx, ty])[0]*dx_tile*dy_tile  # density * area = probability
        print(f"True values: ({tx:.2f}, {ty:.2f}) for sample {idx + 1}")
        kde_at_test.append(val)
        product *= val
        avg += val

    fig.colorbar(contours, ax=ax, label='Density')

    # ------- 4. Final plot cosmetics -----------------------------------------------
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Five samples with their Gaussian KDE contours")
    ax.legend(fontsize="small", frameon=False)
    plt.tight_layout()
    plt.show()

    # ------- 5. Report the KDE values & their product ------------------------------
    print("KDE values evaluated at each sample’s test (x, y):")
    for i, v in enumerate(kde_at_test, 1):
        print(f"  Sample {i}: {v:.6e}")
    print(f"\ntelescope={telescope}, feature={feature}, Product of the five KDE values: {product:.6e}")
    print(f"\ntelescope={telescope}, feature={feature}, Avg of the five KDE values: {avg:.6e}")