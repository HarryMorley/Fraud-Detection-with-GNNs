import pandas as pd
import numpy as np



def chronological_purged_cv(
    df: pd.DataFrame,
    ts_col: str = "Timestamp",
    n_splits: int = 5,
    *,
    test_fraction: float = 0.15,
    test_rows: Optional[int] = None,
    embargo: pd.Timedelta = pd.Timedelta(0),
    min_test_rows: int = 1,
    label_col: Optional[str] = None,
    min_test_pos: int = 0,
) -> List[Dict]:
    """
    Chronological CV with **equal-row partitions** (by count), not equal time.

    - Sort all rows by time.
    - Split the *sorted index* into ``n_splits`` contiguous blocks with (approximately) the same number of rows.
    - For each block, the test set is the **last `test_rows` rows** of the block. If ``test_rows`` is ``None``,
      it uses ``ceil(test_fraction * block_size)``. A floor of ``min_test_rows`` is enforced.
    - Training set = **all rows with timestamp < (test_start_time - embargo)** across the whole dataset.
      This yields expanding, purged training windows and avoids any future leakage.
    - If ``label_col`` is provided, the function will check the number of positives in each test set and warn
      (or, if ``min_test_pos>0``, it will grow the test set backwards within the block to try to reach the target.
      If it still cannot, it keeps the maximal available block and warns.)

    Returns a list of dicts per fold with indices and date ranges.
    """
    if ts_col not in df.columns:
        raise KeyError(f"{ts_col} not in dataframe")

    d = df[[ts_col]].dropna(subset=[ts_col])
    if d.empty:
        raise ValueError("No non-null timestamps available after parsing.")

    # Global sorted index by time
    order = d[ts_col].sort_values().index
    ts_sorted = df.loc[order, ts_col]

    n = len(order)
    if n_splits <= 0:
        raise ValueError("n_splits must be >= 1")
    if n_splits > n:
        raise ValueError("n_splits cannot exceed number of rows")

    # Row-quantile boundaries
    cuts = np.linspace(0, n, n_splits + 1, dtype=int)

    folds: List[Dict] = []
    for i in range(n_splits):
        lo, hi = cuts[i], cuts[i + 1]
        part_idx = order[lo:hi]
        if len(part_idx) == 0:
            continue

        part_start_time = df.loc[part_idx[0], ts_col]
        part_end_time   = df.loc[part_idx[-1], ts_col]

        block_size = len(part_idx)
        desired_test = test_rows if test_rows is not None else int(np.ceil(test_fraction * block_size))
        desired_test = max(desired_test, min_test_rows)
        desired_test = min(desired_test, block_size)  # cannot exceed block size

        # Select last `desired_test` rows of the block as test
        test_idx = part_idx[-desired_test:]
        test_start_time = df.loc[test_idx[0], ts_col]
        test_end_time   = df.loc[test_idx[-1], ts_col]

        # If label constraint provided, try to widen within the block
        if label_col is not None and min_test_pos > 0 and label_col in df.columns:
            pos_count = int(df.loc[test_idx, label_col].sum())
            if pos_count < min_test_pos:
                # widen backwards inside the block
                need = min_test_pos - pos_count
                # grow by chunks of 5% of the block or at least 100 rows each step
                step = max(100, block_size // 20)
                take = desired_test
                while pos_count < min_test_pos and take < block_size:
                    take = min(block_size, take + step)
                    test_idx = part_idx[-take:]
                    pos_count = int(df.loc[test_idx, label_col].sum())
                test_start_time = df.loc[test_idx[0], ts_col]
                test_end_time   = df.loc[test_idx[-1], ts_col]
                if pos_count < min_test_pos:
                    print(
                        f"Warning: fold {i+1} reached only {pos_count} positives (< {min_test_pos}) "
                        f"with entire block of {block_size} rows."
                    )
        elif label_col is not None and label_col not in df.columns:
            print(f"Warning: label_col '{label_col}' not found; skipping min_test_pos checks.")

        # Training: all rows strictly before (test_start_time - embargo)
        train_cut_time = test_start_time - embargo
        train_mask = ts_sorted < train_cut_time
        train_idx = ts_sorted.index[train_mask]

        folds.append(
            dict(
                fold=i + 1,
                train_idx=train_idx,
                test_idx=test_idx,
                part_start=part_start_time,
                part_end=part_end_time,
                test_start=test_start_time,
                test_end=test_end_time,
                train_end=train_cut_time,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )

    if len(folds) < n_splits:
        print(
            f"Warning: produced {len(folds)} folds (requested {n_splits}). Some partitions were empty."
        )

    return folds







class FeatureEngineer:
    """
    Fit on a training DataFrame to discover:
      - top‐k high‐risk sender & receiver banks
      - top‐k high‐risk currencies
      - all payment‐format categories
    Then transform any DataFrame by:
      - log(Amount Received)
      - fee flag & fee pct
      - cyclical time features
      - internal/external transfer flags
      - one‐hot Payment Format
      - binary high‐risk flags (banks & currencies)
    """
    def __init__(self,
                 top_k_banks: int = 10,
                 top_k_currencies: int = 6,
                 target_col: str = "Is Laundering",
                 ts_col: str = "Timestamp"):
        self.top_k_banks = top_k_banks
        self.top_k_currencies = top_k_currencies
        self.target_col = target_col
        self.ts_col = ts_col

    def fit(self, df: pd.DataFrame):
        # 1) Ensure timestamp
        df = df.copy()
        df[self.ts_col] = pd.to_datetime(df[self.ts_col], 
                                         format="%Y/%m/%d %H:%M",
                                         errors="coerce")

        # 2) Learn high‐risk banks
        sender_rates = (
            df.groupby("From Bank")[self.target_col]
              .mean()
              .nlargest(self.top_k_banks)
        )
        receiver_rates = (
            df.groupby("To Bank")[self.target_col]
              .mean()
              .nlargest(self.top_k_banks)
        )
        self.high_risk_senders   = set(sender_rates.index)
        self.high_risk_receivers = set(receiver_rates.index)

        # 3) Learn high‐risk currencies (combine receive+payment)
        cur_df = pd.concat([
            df[["Receiving Currency", self.target_col]]
               .rename(columns={"Receiving Currency":"Currency"}),
            df[["Payment Currency",   self.target_col]]
               .rename(columns={"Payment Currency":  "Currency"})
        ])
        currency_rates = (
            cur_df.groupby("Currency")[self.target_col]
                  .mean()
                  .nlargest(self.top_k_currencies)
        )
        self.high_risk_currencies = set(currency_rates.index)

        # 4) Remember all seen payment formats
        self.payment_formats = df["Payment Format"].dropna().unique().tolist()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # — 1) Timestamp → cyclical features
        df[self.ts_col] = pd.to_datetime(df[self.ts_col], 
                                         format="%Y/%m/%d %H:%M",
                                         errors="coerce")
        h = df[self.ts_col].dt.hour.fillna(0).astype(int)
        dow = df[self.ts_col].dt.dayofweek.fillna(0).astype(int)
        df["tod_sin"] = np.sin(2 * np.pi * h / 24)
        df["tod_cos"] = np.cos(2 * np.pi * h / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # — 2) Amount features
        df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
        df["log_amount_received"] = np.log1p(df["Amount Received"])
        # fee flag & fee pct
        df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
        df["fee_flag"] = (df["Amount Paid"] != df["Amount Received"]).astype(int)
        df["fee_pct"]  = ((df["Amount Paid"] - df["Amount Received"]) 
                          / df["Amount Received"]
                         ).fillna(0)

        # — 3) Transfer flags
        df["internal_transfer"] = (df["From Bank"] == df["To Bank"]).astype(int)
        df["external_transfer"] = (df["From Bank"] != df["To Bank"]).astype(int)

        # — 4) High-risk bank flags
        df["high_risk_sender"] = df["From Bank"].isin(self.high_risk_senders).astype(int)
        df["high_risk_receiver"] = df["To Bank"].isin(self.high_risk_receivers).astype(int)

        # — 5) High-risk currency flag
        # Check both Receive & Payment
        df["high_risk_currency"] = (
            df["Receiving Currency"].isin(self.high_risk_currencies) |
            df["Payment Currency"].isin(self.high_risk_currencies)
        ).astype(int)

        # — 6) One-hot Payment Format (guaranteed same columns as training)
        for fmt in self.payment_formats:
            col = f"pf_{fmt}"
            df[col] = (df["Payment Format"] == fmt).astype(int)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
