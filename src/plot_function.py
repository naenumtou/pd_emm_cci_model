
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, norm
from scipy.special import expit

# Plot waterfall exclusion
def plot_exclusion(
    log: list
) -> None:
    
    """
    Plot waterfall exclusion.

    Description:
        Plot count of waterfall exclusion on each criteria.

    Args:
        log (list): List of excluded counts.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """

    df_plot = (
        pd.DataFrame(log, columns = ['Criteria', 'Before', 'After'])
        .set_index('Criteria')
    )

    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray    
    colors = ['red'] * len(df_plot)
    colors[0] = colorG
    colors[-1] = colorY

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(df_plot.index, df_plot["Before"], color = colors)
    ax.bar(df_plot.index, df_plot["After"], color = 'white')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])
    ax.set_title("Waterfall exclusion")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Number of observation")
    ax.tick_params(axis = "x", rotation = 90)
    plt.tight_layout()

    return plt.show()

# Plot matrix
def plot_avg_matrix(
    data: pd.DataFrame
) -> None:
    
    """
    Plot average migration matrix.

    Description:
        Plot average migration matrix for model.

    Args:
        data (pd.DataFrame): Average migration matrix.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """
    
    fig, axs = plt.subplots(figsize = (10, 6))
    fig.suptitle("Average migration matrix")
    sns.heatmap(
       data, annot = True,
       fmt = '.2%', cmap = 'RdYlGn_r', cbar = False,
       ax = axs
    )
    
    return plt.show()

# Plot series dependence variable
def plot_dep_var(
    y: pd.Series,
    y_target: pd.Series,
    target_label: str
) -> None:
    
    """
    Plot dependence variables for regression model.

    Description:
        Showing the historical dependence variables (logit, CF, CCI) for regression model.

    Args:
        y (pd.Series)           : The actual monthly ODR.
        y_target (pd.Series)    : The transformed dependence variable.
        target_label (str)      : The method name. E.g., "Logit", "CF", "CCI".

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - This function will be called in the src.regression_model in prepare_training_set().
    """
    
    fig, ax = plt.subplots(figsize = (10, 6))
    fig.suptitle("Historical dependence variable")
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray

    ax.plot(y_target, color = colorY, linewidth = 2, label = target_label)
    ax.set_yticklabels([f"{y:.4f}" for y in ax.get_yticks()])

    if target_label != "CCI":
        ax_r = ax.twinx()
        ax_r.plot(y, color = colorG, linestyle = "--", linewidth = 2, label = "ODR")
        ax_r.set_yticklabels([f"{y * 100:.2f}%" for y in ax_r.get_yticks()])
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc = "upper right")
    else:
        ax.legend(loc = "upper right")
    plt.tight_layout()
    
    return fig

# Plot univariate result
def plot_univariate(
    df: pd.DataFrame,
    p_threshold: float,
    r2_threshold: float
) -> None:

    """
    Plot univariate result.

    Description:
        Showing the univariate analysis result by plotting R-Square against log10(p-value).
        The red highlights of high R-Sqaure and p-value are shown the wrong intuitive sign.

    Args:
        df (pd.DataFrame)       : The summary table of univariate analysis.
        p_threshold (float)     : p-value threshold.
        r2_threshold (float)    : R-Square threshold.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - This function will be called in the src.regression_model in single_regression().
    """

    palette = {
        "Pass all 3": "#2ecc71",
        "Wrong sign": "#e74c3c",
        "R² ≤ 50%": "#f39c12",
        "Not significant": "#95a5a6"
    }
    order = ["Not significant", "R² ≤ 50%", "Wrong sign", "Pass all 3"]
    fig, ax = plt.subplots(figsize = (10, 6))
    
    for cat in order:
        sub = df[df["category"] == cat]
        ax.scatter(
            sub["neg_log_p"], sub["r2"],
            c = palette[cat], label = cat,
            alpha = 0.6 if cat != "Pass all 3" else 1.0,
            s = 18 if cat != "Pass all 3" else 30,
            linewidths = 1
        )

    ax.axvline(
        -np.log10(p_threshold), color = "Red", lw = 0.8,
        ls = "--", alpha = 0.6, label = f"p = {p_threshold}"
    )
    ax.axhline(
        r2_threshold, color = "Red", lw = 0.8,
        ls = "--", alpha = 0.6, label = f"R² = {r2_threshold:.0%}"
    )
    ax.set_xlabel("-log10(p-value)")
    ax.set_ylabel("R²")
    ax.set_title("Univariate analysis\n(p-value and R² and Sign)")
    ax.legend(frameon = True, facecolor = 'white', loc = "lower right")
    plt.tight_layout()

    return fig

# Plot multivariate result
def plot_cluster_timeseries(
    X: pd.DataFrame,
    result: pd.DataFrame,
):

    """
    Plot multivariate result.

    Description:
        Showing the multivariate analysis result by plotting Z-Score per cluster.
        The colors highlight of selected time seies MEV(s).

    Args:
        X (pd.DataFrame)        : The transformed MEV(s) Data.
        result (pd.DataFrame)   : The table result from multivariate analysis.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - This function will be called in the src.regression_model in multivariate_selection().
    """
    
    fig, axs = plt.subplots(
        int(result["Cluster"].max() / 3), 3, figsize = (20, 8),
        sharex = True, sharey = True
    )
    axs = axs.ravel()
    fig.suptitle("Multivariate analysis\n(Variables clustering)")
    colorG = '#808080' #Set color theme --> Gray

    clusters = sorted(result["Cluster"].unique())

    for ax, cluster in zip(axs, clusters):
        var_list = result.loc[result["Cluster"] == cluster, "Variable"].tolist()
        pass_list = result.loc[(result["Cluster"] == cluster) & (result["pass"] == True), "Variable"].tolist()

        data_z = zscore(X[var_list])
        ax.plot(X.index, data_z, color = colorG, alpha = 0.5, linewidth = 0.5)
        for pass_var in pass_list:
            pass_z = zscore(X[pass_var])      
            ax.plot(X.index, pass_z, linewidth = 2)

        ax.set_yticklabels([f"{y:.4f}" for y in ax.get_yticks()])
        ax.set_title(f"Cluster - {cluster}")

    # Close un-used subplot
    for ax in axs[len(clusters):]:
        ax.set_visible(False)

    fig.supylabel("Z-Score")
    plt.tight_layout(rect = (0.01, 0, 1, 1))

    return fig

# Plot back-testing
def plot_backtest(
    y_train: pd.Series,
    model: dict,
    model_name: str,
    model_method: str,
    std_params: pd.DataFrame = None
) -> None:
    
    """
    Plot model back-testing.

    Description:
        Showing the historical time series between actual and predicted from the model.

    Args:
        y_train (pd.Series)                     : The dependence variable target data (Logit, CF or CCI).
        model (dict)                            : The dictionary of all candidate models.
                                                Keys are the candidate models name.
                                                Values are trained model output from sm.OLS().fit().
                                                {keys: values} --> {str: callable}
        model_name (str)                        : The random model name for showing the back-testing.
        model_method (str)                      : Name of the regression method. The function is plotted;
                                                1) model_method = "Logit" --> %ODR vs %predicted ODR.
                                                2) model_method = "CF" --> Inverse CF and compute %ODR vs %predicted ODR.
                                                3) model_method = "CCI" --> CCI vs predicted CCI.
                                                If model_method = "CCI", the std_params must input as pd.DataFrame
        std_params (pd.DataFrame, optional)     : The data table contained standardisation parameters.
                                                If None, the model_method must be "Logit" or "CCI".

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """
    
    # Select model for the test
    model = model[model_name]

    if model_method == "CCI":
        y_pred = model.predict()
        y_true = y_train.copy()

    if model_method == "Logit":
        y_pred = expit(model.predict())
        y_true = expit(y_train)

    if model_method == "CF":
        mean = std_params.loc["Dependence_Variable", "mean"]
        std = std_params.loc["Dependence_Variable", "std"]
        y_pred = pd.Series(norm.cdf((model.predict()) * std + mean))
        y_true = pd.Series(norm.cdf(y_train * std + mean), index = y_train.index)

    sd = y_pred.std()
    upper = y_pred + 2 * sd
    lower = y_pred - 2 * sd

    # Plot    
    fig, ax = plt.subplots(figsize = (10, 6))
    fig.suptitle(f"Back-testing: {model_name}")
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray

    ax.plot(y_true.index, y_true, color = colorY, linewidth = 2, label = "Actual")
    ax.plot(y_true.index, y_pred, color = colorG, linewidth = 2, label = "Predicted")
    ax.plot(
        y_true.index, upper, color = colorG,linestyle = "--",
        linewidth = 1, alpha = 0.6, label = "+2S.D."
    )
    ax.plot(
        y_true.index, lower, color = colorG, linestyle = "--",
        linewidth = 1, alpha = 0.6, label = "-2S.D."
    )
    ax.plot([], [], ' ', label = f"Adj. R-Square: {model.rsquared_adj * 100:.2f}%")
    
    if model_method != "CCI":
        ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
    else:
        ax.set_yticklabels([f"{y:.4f}" for y in ax.get_yticks()])
        
    ax.legend(frameon = True, facecolor = 'white', loc = "upper right")
    plt.tight_layout()
    
    return plt.show()
