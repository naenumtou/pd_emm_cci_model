
import pandas as pd
import numpy as np

# Matrix multiplication
def matrix_lifetime(
    avg_matrix: pd.DataFrame,
    pred_matrix: np.ndarray,
    years: int,
    mean_reversion: bool = False
) -> pd.DataFrame:
    
    """
    Lifetime PD by matrix multiplication.

    Description:
        Lifetime PD by matrix multiplication projects credit risk over
        the given lifetime period. By multiplying a one-year migration matrix
        by next n-year migration matrix recursively, analysts build a 
        cumulative PD Term structure.
        
        This is the Markov Chain method estimates yearly migration to default.
        The standard migration model assumes the probability of migration depends
        on the current state and remaining state over the time.

        Point-In-Time (PIT) Adjustment: Since 1-year historic matrix reflects
        the average (Through-The-Cycle) behavior, IFRS 9 requires adjusting
        the matrix to reflect current and forward-looking macroeconomic conditions
        before doing matrix multiplication. The predicted migration (PiT) is used
        for a given year matrix multiplication.

        Mean Reversion: Because macroeconomic forecasts (e.g., GDP, unemployment)
        become unreliable after 3 to 5 years, the adjusted matrices are typically
        forced to revert to historical, long-term averages in later periodsใ

    Args:
        avg_matrix (pd.DataFrame)   : Average matrix.
        pred_matrix (np.ndarray)    : Perdicted matrix.
        years (int)                 : Years for lifetime PD Creation.
        mean_reversion (bool)       : Option for year beyond prediction. Either using latest migration or average migration.

    Returns:
        pd.DataFrame: n-years cumulative PD Term structure.

    Notes:
        - N/A.

    """
    
    # Symmetry matrix
    absorb_row = np.zeros((pred_matrix.shape[0], 1, pred_matrix.shape[2]))
    absorb_row[:, 0, -1] = 1 #Absorbing state

    # Append to fitted matrix
    pred_matrix_full = np.concatenate(
        [pred_matrix, absorb_row],
        axis = 1
    )  #Shape --> (n, 5, 5)

    # Matrix multiplication
    results = [pred_matrix_full[0]] #Year 1
    latest_matrix = pred_matrix_full[-1]

    for t in range(1, years):
        if t < pred_matrix.shape[0]:
            base = pred_matrix_full[t]

        else:
            # Year beyond prediction
            if mean_reversion:
                base = avg_matrix.values #Mean reversion fallback to average migration
            else:
                base = latest_matrix #Using latest constant matrix
            
        results.append(results[-1] @ base) 

    # Lifetime matrix
    lifetime_matrix = np.stack(results)  #Shape --> (n, 5, 5)
    lifetime_pd = lifetime_matrix[:, :, -1] #Extract last column --> default

    # To DataFrame
    cum_pd = pd.DataFrame(
        lifetime_pd[:, :-1], #Without absorbing state
        columns = avg_matrix.index[:-1]
    ).T
 
    return cum_pd
