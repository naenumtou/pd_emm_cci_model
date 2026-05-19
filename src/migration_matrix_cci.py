
import warnings
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import minimize

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
# Monthly matrix
def _monthly_matrix(
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
        df (pd.DataFrame)   : Input data table as the long format, counted observation.
        date_col (str)      : Period column name.
        del1_col (str)      : Initial observation column name.
        del12_col (str)     : Performance column name.

    Returns:
        np.ndarray: Monthly migration matrix (m, n - 1, n) shape.

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
    monthly = monthly_migration.div(
            monthly_migration.sum(axis = 1), axis = 0
        ).fillna(0)

    # Reshape (m, n - 1, n)
    m = df[date_col].nunique()
    n = len(states)
    
    return monthly.values.reshape(m, n - 1, n)

# Number of observations on monthly
def _obs_array(
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
        df (pd.DataFrame)    : Input data table as the long format, counted observation.
        date_col (str)       : Period column name.
        del1_col (str)       : Initial observation column name.

    Returns:
        np.ndarray: Monthly observations count (m, n - 1) shape.

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
    
    # Reshape (m, n - 1)
    m = df[date_col].nunique()
    n = len(states)
    
    return obs_n.values.reshape(m, n - 1)

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
        df (pd.DataFrame)   : Input data table as the long format, counted observation.
        del1_col (str)      : Initial observation column name.
        del12_col (str)     : Performance column name.

    Returns:
        pd.DataFrame: Average migration matrix (n, n) shape.

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
        df (pd.DataFrame): Input average migration matrix as (n, n) shape.

    Returns:
        np.ndarray: Upper threshold for average migration matrix (n - 1, n) shape.

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

# Fitted CDF
def fitted_cdf(
    ppf_matrix: np.ndarray
) -> np.ndarray:

    """
    Convert inverse of the cumulative distribution function.

    Description:
        Convert inverse of the cumulative distribution function to
        cumulative distribution function. Then, transform it back to
        fitted migration rate.

    Args:
        ppf_matrix (np.ndarray): Input inverse of the cumulative distribution function.

    Returns:
        np.ndarray: Migration rate array.

    Notes:
        - If a single matrix --> shape (n - 1, n).
        - If a more than one matrix --> shape (m, n - 1, n).
    """

    # Compute CDF for entire matrix at once
    cdf = norm.cdf(ppf_matrix)
    
    # 2D array
    if cdf.ndim == 2:
        cdf_sub = cdf[:, 1:]
        
        # Build fitted matrix
        fitted_matrix = np.column_stack(
            [
                1 - cdf_sub[:, 0],
                cdf_sub[:, :-1] - cdf_sub[:, 1:],
                cdf_sub[:, -1]
            ]
        )
    
    # 3D array
    elif cdf.ndim == 3:
        cdf_sub = cdf[:, :, 1:]

        # Build fitted matrix
        fitted_matrix = np.concatenate(
            [
                1 - cdf_sub[:, :, [0]],
                cdf_sub[:, :, :-1] - cdf_sub[:, :, 1:],
                cdf_sub[:, :, [-1]]
            ],
            axis = 2
        )

    else:
        raise ValueError("ppf matrix must be 2D or 3D")
        
    return fitted_matrix

# CCI
def credit_cycle_index(
    x: np.ndarray,
    df: pd.DataFrame,
    average_matrix: pd.DataFrame,
    date_col: str,
    del1_col: str,
    del12_col: str
) -> float:
    
    """
    Credit Cycle Index.

    Description:
        Getting the average observed migration matrix, upper bounds are computed
        to derive the fitted transmigrationition matrix. The bounds are determined
        by obtaining the inverse normal value based on the migration rates.
        
        Construct fitted migration matrix to fit observed migration matrix to
        average observed migration matrix.

        As the migration rates are fitted by computing the deviation from a base matrix,
        it is then able to compute an error term that represents deviation for pre-defined Rho.

        The model is optimized by minimizing the error term for a pre-defined Rho.
        The process is iterated multiple times until CCI or Z-Index obtains a variance of 1.
        The Rho can be interpreted as the weightage of the relationship between the
        migration rate and CCI or Z-Index at time t.

    Args:
        x (np.ndarray)                  : n-periods of random generated numbers.
                                        The first n positions are random generated CCI by monthly basis.
                                        The last position is random generated Rho.
        df (pd.DataFrame)               : Input data table as the long format, counted observation.
        average_matrix (pd.DataFrame)   : Input of average migration matrix for upper array computation.
        date_col (str)                  : Period column name.
        del1_col (str)                  : Initial observation column name.
        del12_col (str)                 : Performance column name.

    Returns:
        float: The sum of squared error between fitted migration matrix and observed migration matrix.

    Notes:
        - N/A.
    """

    # Parameters
    eps = 1e-12 #Avoid division by zero
    rho = x[-1] #Set the rho at last position

    # Data
    upper_ = upper_threshold(average_matrix)
    monthly_ = _monthly_matrix(df, date_col, del1_col, del12_col)
    obs_ = _obs_array(df, date_col, del1_col)
    
    # Fitting upper
    monthly_fitted = (
        upper_[None, :, :] - (np.sqrt(rho) * x[:-1, None, None]) #Without last position
    ) / np.sqrt(1 - rho)

    # Fitted
    matrix_fitted = fitted_cdf(monthly_fitted)

    # Error
    denominator = np.clip(matrix_fitted * (1 - matrix_fitted), eps, None)
    error = (obs_[:, :, None] * ((monthly_ - matrix_fitted) ** 2)) / denominator
    
    return np.sum(error)

# Constraint CCI
def constraint_std(
    x: np.ndarray
) -> float:
    
    """
    Constraint CCI.

    Description:
        The constraint function for CCI is obtained a variance of 1.

    Args:
        x (np.ndarray): n-periods of random generated numbers.
                        The first n positions are random generated CCI by monthly basis.
                        The last position is random generated Rho.

    Returns:
        float: Variance of CCI equal to 1.

    Notes:
        - N/A.
    """

    return np.std(x[:-1]) - 1

# Constraint Rho
def constraint_corr(
    x: np.ndarray
) -> float:
    
    """
    Constraint Rho.

    Description:
        The constraint function for Rho is between 0% - 100%.

    Args:
        x (np.ndarray): n-periods of random generated numbers.
                        The first n positions are random generated CCI by monthly basis.
                        The last position is random generated Rho.

    Returns:
        float: Estimated Rho bounds (0-1).

    Notes:
        - N/A.
    """

    return 0.999 - np.abs(x[-1])

# Optimization CCI
def find_cci(
    df: pd.DataFrame,
    average_matrix: pd.DataFrame,
    date_col: str,
    del1_col: str,
    del12_col: str,
    const_cci: Callable = constraint_std,
    const_rho: Callable = constraint_corr
) -> OptimizeResult:
    
    """
    Optimization CCI.

    Description:
        The optimization for CCI.

    Args:
        df (pd.DataFrame)               : Input data table as the long format, counted observation.
        average_matrix (pd.DataFrame)   : Input of average migration matrix for upper array computation.
        date_col (str)                  : Period column name.
        del1_col (str)                  : Initial observation column name.
        del12_col (str)                 : Performance column name.
        const_cci (Callable             : Constraint function for CCI.
        const_rho (Callable             : Constraint function for Rho.

    Returns:
        OptimizeResult: The optimization object contained CCI and Rho.

    Notes:
        - N/A.
    """

    # Initial CCI --> random
    init0 = np.hstack(
        (
            np.random.randn(df[date_col].nunique()),
            0.0001
        )
    )

    # Constraints functions
    constraints = [
        {'type': 'eq', 'fun': const_cci}, #std of CCI = 1
        {'type': 'ineq', 'fun': const_rho} #|rho| < 1
    ]

    # Boundaries --> None for CCI, 0-1 for Rho
    bounds = [(None, None)] * (len(init0) - 1) + [(1e-6, 0.999)]

    # Find CCI
    result = minimize(
        fun = credit_cycle_index,
        x0 = init0,
        args = (df, average_matrix, date_col, del1_col, del12_col),
        method  = "SLSQP",
        constraints = constraints,
        bounds = bounds,
        options = {'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 100_000}
    )
    
    return result
