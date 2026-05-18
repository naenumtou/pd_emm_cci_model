
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function

# Average migration matrix
def avg_matrix(
    df: pd.DataFrame,
    del1_col: str,
    del12_col: str
) -> pd.DataFrame:
    
    """
    Average migration matrix.

    Description:
        Compute the average migration matrix. The migration matrix simply captures the rate
        of migration between different buckets for the accounts observed at any observation
        months to performance months. The rates are computing by counting the accounts
        migration divided by initial accounts at observed.

    Args:
        df (pd.DataFrame)         : Input data table as the long format, counted observation.
        del1_col (pd.Series)      : Initial observation period.
        del12_col (pd.Series)     : Performance period.

    Returns:
        pd.DataFrame: Average migration matrix (n x n) shape.

    Notes:
        - N/A.
    """

    average_matrix = (
        df.groupby([del1_col, del12_col], as_index = False)["n"]
        .sum()
    )

    average_matrix["total"] = average_matrix.groupby(del1_col)["n"].transform("sum")
    average_matrix["transition_rate"] = average_matrix["n"] / average_matrix["total"].replace(0, np.nan) #Avoid division by zero
    average_matrix["transition_rate"] = (
        average_matrix["transition_rate"] /
        average_matrix.groupby(del1_col)["transition_rate"].transform("sum")
    ) #Ensure rows sum to 1

    # Pivot into matrix form
    transition_matrix = average_matrix.pivot(
        index = del1_col,
        columns = del12_col,
        values = "transition_rate"
    ).fillna(0)

    return transition_matrix

# Upper threshold
def upper_threshold(
    df: pd.DataFrame
) -> np.ndarray:

    """
    Upper threshold for average migration matrix.

    Description:
        The migration matrix adopts a concept that the migration are driven
        by a standard normally distributed of variable. Therefore, instead of
        describing migration behaviour through transition rates, it is described
        through a set of threshold (e.g., binning of a standard normal distribution).
        The migration rate equals to the area enclosed by the boundaries of the bin
        and the density function.

    Args:
        df (pd.DataFrame): Input average migration matrix as (n x n) shape.

    Returns:
        np.ndarray: Upper threshold for average migration matrix (n - 1 x n) shape.

    Notes:
        - It needs to remove the last row as absorbing state.
    """

    upper = norm.ppf(
        1 - df.cumsum(axis = 1)
    )
    upper = np.roll(
        upper,
        shift = 1,
        axis = 1
    )

    return upper[:-1] #Remove the last row
