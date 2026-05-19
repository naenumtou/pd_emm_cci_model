
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
        del1_col (pd.Series)      : Initial observation column name.
        del12_col (pd.Series)     : Performance column name.

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

# Monthly matrix
def monthly_matrix(
    df: pd.DataFrame,
    date_col: str,
    del1_col: str,
    del12_col: str
) -> np.ndarray:
    
    """
    Monthly migration matrix.

    Description:
        Compute the monthly migration matrix. The migration matrix simply captures the rate
        of migration between different buckets for the accounts observed at any observation
        months to performance months. The rates are computing by counting the accounts
        migration divided by initial accounts at observed.

    Args:
        df (pd.DataFrame)         : Input data table as the long format, counted observation.
        date_col (pd.Series)      : Period column name.
        del1_col (pd.Series)      : Initial observation column name.
        del12_col (pd.Series)     : Performance column name.

    Returns:
        np.ndarray: Monthly migration matrix (m x n - 1 x n) shape.

    Notes:
        - The same computation as average migration matrix but it is in monthly basis.
        - The monthly migration needs to be a symmetry matrix due to some month may not
          have full observations count. For example, lack of accounts migrated from 1 to 3.
          If the table does not symmetry, the matrix is wrong computation.
        - It needs to remove the last row on each month as absorbing state.
    """

    # Define all states
    states = np.arange(df[[del1_col, del12_col]].max().max() + 1) #Migration matrix always defines maximum value at worst

    # Aggregate first
    df_agg = df.groupby([date_col, del1_col, del12_col])["n"].sum()

    # Create full index grid
    full_index = pd.MultiIndex.from_product(
        [df[date_col].unique(), states, states],
        names = [date_col, del1_col, del12_col]
    )
    
    # Reindex to enforce symmetry
    df_full = df_agg.reindex(full_index, fill_value = 0)

    # Pivot
    monthly_migration = df_full.unstack(del12_col, fill_value = 0)

    # Remove del = max state (default)
    monthly_migration = monthly_migration.loc[
        monthly_migration.index.get_level_values(del1_col) != states[-1]
    ]

    # Normalize (to percent)
    return monthly_migration.div(
        monthly_migration.sum(axis = 1), axis = 0
    ).fillna(0).values

# Number of observations on monthly
def obs_array(
    df: pd.DataFrame,
    date_col: str,
    del1_col: str
) -> np.ndarray:

    """
    Monthly observations count.

    Description:
        Compute the monthly observations initial count. The count is for the cost
        function for minimise the error during CCI Optimisation process.

    Args:
        df (pd.DataFrame)         : Input data table as the long format, counted observation.
        date_col (pd.Series)      : Period column name.
        del1_col (pd.Series)      : Initial observation column name.

    Returns:
        np.ndarray: Monthly observations count (m x 1) shape.

    Notes:
        - The monthly observations count needs to be a symmetry table due to some month may not
          have full observations count. For example, lack of accounts migrated from 1 to 3.
          If the table does not symmetry, the matrix is wrong computation.
        - It needs to remove the last row on each month as absorbing state.
    """

    # Define all states
    states = np.arange(df[del1_col].max() + 1) #Migration matrix always defines maximum value at worst

    # Aggregate first
    df_agg = df.groupby([date_col, del1_col])["n"].sum()

    # Create full index grid
    full_index = pd.MultiIndex.from_product(
        [df[date_col].unique(), states],
        names = [date_col, del1_col]
    )

    # Reindex to enforce symmetry
    obs_n = df_agg.reindex(full_index, fill_value = 0)
    
    # Remove del = max state (default)
    obs_n = obs_n.loc[
        obs_n.index.get_level_values(del1_col) != states[-1]
    ]
    
    return obs_n.values
