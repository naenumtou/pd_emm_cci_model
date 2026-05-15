
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper functions
# Lag-n columns
def _lag_cols(
    base: str,
    n: int
) -> list[str]:

    """
    Lagging columns functions.

    Description:
        Lagging columns only used for the calculation.

    Args:
        base (str)  : Columns name for lagging.
        n (int)     : Window to lag the columns.

    Returns:
        List: List of lag column names.

    Notes:
        - N/A.
    """

    return [f"{base}{i}" for i in range(1, n + 1)]

# Forward performance windows until lifetime
def ever_default(
    df: pd.DataFrame,
    id_col: str,
    period_col: str,
    default_col: str,
    default_flag: int = 4,
    n_lags: int = 12
) -> pd.DataFrame:
    
    """
    Sort by primary key and period and create forward-1 until forward-n
    columns for target. Uses a single groupby().transform(lambda).

    Description:
        The n-lags of column features are created by primary key. 

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        period_col (str)    : Period key for sorting.
        default_col (str)   : Default column that target for modeling.
        default_flag (int)  : Default value that target for modeling.
        n_lags (int)        : Defined n-lags for ever default creation.

    Returns:
        pd.DataFrame: DataFrame with sorted, forward performance columns
                      and ever default flag appended.

    Notes:
        - Using the worst status both of good or bas observataion assumption
          for migration matrix creation.
    """

    print("=== Processing ===\n[Sort, Forward performance windows and Ever default]")

    df = df.sort_values(by = [id_col, period_col]).copy()

    # Forward performance windows until lifetime
    grouped = df.groupby(id_col)[default_col]
    shifted = {
        f"{default_col}{i}": grouped.shift(-i).astype(np.float16)
        for i in range(1, n_lags + 1) #(Exclusive) Need to +1 because need 12 months to observe
    }
    df = df.assign(**shifted)

    # Ever default flag
    cols = _lag_cols(default_col, n_lags)
    window = df[cols]
    df[f"del_ever_{n_lags}"] = np.where(
        window.eq(default_flag).any(axis = 1) | df[default_col].eq(default_flag),
        default_flag,
        window.max(axis = 1)
    )

    return df

# Count migration
def count_migration(
    period_col: pd.Series,
    del1_col: pd.Series,
    del12_col: pd.Series
) -> None:

    """
    Migration count.

    Description:
        Compute raw summary of migration in the monthly basis.

    Args:
        period_col (pd.Series)    : Period key for summary.
        del1_col (pd.Series)      : Initial observation period.
        del12_col (pd.Series)     : Performance period.

    Returns:
        Parquet file: Storaged file as .parquet format in '../data/processed'.

    Notes:
        - This is a input for further CCI Modeling steps.
    """

    print("=== Processing ===\n[Migration count]")
    
    filename = "migration_count"

    df = pd.DataFrame(
        {
            "date": period_col,
            "del": del1_col,
            'del12': del12_col
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["del"] = df["del"].astype(int)
    df["del12"] = df["del12"].astype(int)

    agg = {
        "n": ("del", "size")
    }

    migrate = df.groupby(["date", "del", "del12"], as_index = False).agg(**agg)
    migrate.to_parquet(
        f"../data/processed/{filename}.parquet",
        engine = 'pyarrow'
        )

    return print(f"[INFO]: Export - '..data/processed/{filename}.parquet'")
