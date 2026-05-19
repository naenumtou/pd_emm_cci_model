
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot CCI
def plot_cci(
    data: pd.DataFrame,
    rho: float
) -> None:
    
    """
    Plot estimated CCI.

    Description:
        Plot estimated CCI for model.

    Args:
        data (pd.DataFrame) : Historical estimated CCI.
        rho (float)         : Estimated Rho.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """
    
    fig, axs = plt.subplots(figsize = (10, 6))
    fig.suptitle("Historical CCI")
    colorY = '#ffd500' #Set color theme --> Yellow
    axs.plot(data.index, data, color = colorY, linewidth = 2)
    axs.plot([], [], ' ', label = f'Rho: {rho:.4f}')
    axs.set_yticklabels([f"{y:.4f}" for y in axs.get_yticks()])
    axs.legend(frameon = True, facecolor = 'white')
    plt.tight_layout()

    return plt.show()
