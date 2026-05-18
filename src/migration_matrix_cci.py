
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
# Upper threshold


# Average migration matrix
def avg_matrix(
    df: pd.DataFrame,
    del1_col: str,
    del12_col: str
) -> pd.DataFrame:
    
    """
    Average migration matrix.

    Description:
        Compute the average migration matrix.

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



