import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar
import os
import seaborn as sns
import matplotlib.dates as mdates

def optimize_lambda(data):
    """Finds the lambda that minimizes the RMSE of the Nelson-Siegel fit."""
    tau = np.array(data.columns.astype(float))
    yields = data.values

    def objective_lambda(l):
        l1 = np.ones_like(tau)
        l2 = (1 - np.exp(-l * tau)) / (l * tau)
        l3 = l2 - np.exp(-l * tau)
        X = np.column_stack([l1, l2, l3])
        betas = np.linalg.lstsq(X, yields.T, rcond=None)[0]
        fitted = (X @ betas).T
        return np.sqrt(np.mean((yields - fitted)**2))

    print("Optimizing Lambda...")
    res = minimize_scalar(objective_lambda, bounds=(0.01, 2.0), method='bounded')
    return res.x

def fit_data(data, lambda_=0.0609):
    yields = data.values
    dates = pd.to_datetime(data.index, format='%Y-%m-%d')
    tau = np.array(data.columns.astype(float))

    l1 = np.ones_like(tau)
    l2 = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
    l3 = l2 - np.exp(-lambda_ * tau)

    # Use fit_intercept=False for pure factor estimation
    model = LinearRegression(fit_intercept=False)
    X = np.column_stack([l1, l2, l3])

    betas = np.array([model.fit(X, y).coef_ for y in yields])
    betas_df = pd.DataFrame(betas, columns=["beta_1", "beta_2", "beta_3"])
    betas_df["Date"] = dates

    yields_fitted = np.matmul(betas, X.T)
    yields_fitted_df = pd.DataFrame(yields_fitted)

    # Empirical Proxies
    level = data[10]
    slope = data[0.25] - data[10] # NS convention: Short - Long
    curvature = 2*data[2] - (data[10] + data[0.25])

    print("\n--- Factor Interpretation Regression ---")
    for label, target in {"Slope": slope, "Curvature": curvature}.items():
        X_reg = betas_df[["beta_2", "beta_3"]]
        model.fit(X_reg, target)
        print(f"{label} Proxy ≈ {model.coef_[0]:.4f}*β2 + {model.coef_[1]:.4f}*β3 (R²: {model.score(X_reg, target):.2f})")
    
    return [yields_fitted_df, betas_df, None, X]

def get_data(file_path):
    print(f"Loading data from {file_path}...")
    data = pd.read_excel(file_path).dropna()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    return data

def plot_factor_time_series(betas_df, data):
    """
    Reproduces Figure 5: Normalized time series of factors.
    Uses Z-scores to handle scale differences between proxies and betas.
    """
    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    dates = pd.to_datetime(betas_df["Date"]).values
    
    # Empirical Proxies (Calculated as per conventions)
    emp_level = data[10].values
    emp_slope = (data[0.25] - data[10]).values
    emp_curv = (2 * data[2] - (data[10] + data[0.25])).values
    
    # Estimated Factors
    b_factors = [betas_df["beta_1"].values, betas_df["beta_2"].values, betas_df["beta_3"].values]
    p_factors = [emp_level, emp_slope, emp_curv]
    labels = ["Level", "Slope", "Curvature"]

    for i in range(3):
        # Normalize both to Z-scores for visual comparison
        beta_norm = (b_factors[i] - np.mean(b_factors[i])) / np.std(b_factors[i])
        proxy_norm = (p_factors[i] - np.mean(p_factors[i])) / np.std(p_factors[i])

        axes[i].plot(dates, beta_norm, color='black', linewidth=1.8, label=f'β{i+1} (Estimated)')
        axes[i].plot(dates, proxy_norm, color='black', linewidth=0.6, alpha=0.7, label='Observed')
        
        axes[i].set_ylabel(labels[i], fontweight='bold', fontsize=12)
        axes[i].set_xlim(dates.min(), dates.max())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[i].grid(False)
        sns.despine(ax=axes[i])
        axes[i].legend(loc='upper right', frameon=False, fontsize='x-small')

    axes[0].set_title("Time series of factors (standardized)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_data(tau, yields, yields_fitted_df, ns_loadings, betas_df, opt_L):

    sns.set_theme(style="white", context="paper")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Average Yield Curve
    ax_main = fig.add_subplot(gs[:, 0])

    actual_mean = yields.mean(axis=0)
    fitted_mean = yields_fitted_df.mean().values

    sns.scatterplot(x=tau, y=actual_mean, ax=ax_main, color="#d63031", label="Actual Mean", s=60, alpha=0.7, edgecolor='w')
    sns.lineplot(x=tau, y=fitted_mean, ax=ax_main, color="#0984e3", linewidth=3, label="Nelson-Siegel Fit")

    ax_main.set_title("Average Yield Curve Analysis", fontsize=16, fontweight='bold', pad=15)
    ax_main.set_xlabel("Maturity (Years)", fontsize=12)
    ax_main.set_ylabel("Yield (%)", fontsize=12)
    ax_main.legend(fontsize='small', frameon=False)

    # 2. Factor Loadings
    ax_top = fig.add_subplot(gs[0, 1])

    loadings_labels = ["Level_loading", "Slope_loading", "Curvature_loading"]
    palette = ["#2d3436", "#00b894", "#e17055"]

    for i in range(3):
        sns.lineplot(x=tau, y=ns_loadings[:, i], ax=ax_top, label=loadings_labels[i], color=palette[i], linewidth=2)

    ax_top.set_title(f"NSS Loadings (λ = {opt_L:.4f})", fontsize=14, fontweight='bold')
    ax_top.set_ylabel("Loadings")
    ax_top.legend(fontsize='x-small', frameon=False)

    # 3. Betas Over Time
    ax_bot = fig.add_subplot(gs[1, 1])

    sns.lineplot(data=betas_df, x="Date", y="beta_1", ax=ax_bot, color=palette[0], label="β1", linewidth=1)
    sns.lineplot(data=betas_df, x="Date", y="beta_2", ax=ax_bot, color=palette[1], label="β2", linewidth=1)
    sns.lineplot(data=betas_df, x="Date", y="beta_3", ax=ax_bot, color=palette[2], label="β3", linewidth=1)

    ax_bot.set_xlim(betas_df["Date"].min(), betas_df["Date"].max())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax_bot.set_title("Estimated Betas", fontsize=14, fontweight='bold')
    ax_bot.set_ylabel("Parameter Value")
    ax_bot.legend(fontsize='x-small', frameon=False, ncol=3)

    sns.despine()
    plt.show()


# ==========  
# EXECUTION
# ==========
if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(ROOT, "data", "us_treasury.xlsx")
    
    data = get_data(data_path)
    
    if data is not None:
        # Optimize Lambda
        opt_L = optimize_lambda(data)
        
        # Fit with Optimal Lambda
        fitting = fit_data(data, lambda_=opt_L)
        
        betas_ts = fitting[1]

        plot_factor_time_series(betas_ts, data)
        
        fitted_yields = fitting[0]
        betas_ts = fitting[1]
        loadings_mat = fitting[3]
        tau = np.array(data.columns.astype(float))

        plot_data(tau, data.values, fitted_yields, loadings_mat, betas_ts, opt_L)