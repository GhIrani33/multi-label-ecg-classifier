"""Preprocessing utilities for the PTB-XL dataset.

This module provides a production ready preprocessing pipeline for the
PTB-XL electrocardiography dataset.  It implements a configurable data
loading and preprocessing workflow that mirrors the approach typically
used in research notebooks while remaining fully compatible with the rest
of the repository.  The main entry-point remains :func:`preprocess_data`
so that existing training scripts keep working, but the module now also
exposes a set of classes that can be reused in more advanced pipelines or
interactive notebooks.
"""

from __future__ import annotations

import ast
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:  # pragma: no cover - tqdm is optional in some environments
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable, **_):
        """Minimal tqdm replacement used when tqdm is not available."""

        for item in iterable:
            yield item


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass
class PTBXLConfig:
    """Configuration dataclass used throughout the preprocessing pipeline."""

    dataset_path: Path
    output_path: Path

    diagnostic_classes: List[str] = field(
        default_factory=lambda: ["NORM", "MI", "STTC", "CD", "HYP"]
    )
    sampling_rates: Dict[str, int] = field(default_factory=lambda: {"lr": 100, "hr": 500})
    default_sampling_rate: int = 500
    expected_leads: int = 12
    expected_duration: int = 10  # seconds

    age_bounds: Tuple[float, float] = (0.0, 120.0)
    age_encoding_threshold: int = 200

    scaler_type: str = "robust"
    random_seed: int = 42

    use_official_folds: bool = True
    train_folds: List[int] = field(default_factory=lambda: list(range(1, 9)))
    val_fold: int = 9
    test_fold: int = 10

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_path = Path(self.output_path)


class PTBXLDataLoader:
    """Utility class responsible for loading PTB-XL metadata and signals."""

    def __init__(self, config: PTBXLConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset_path

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load PTB-XL metadata from CSV files.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame containing the main PTB-XL records and a DataFrame with
            the SCP statements used for deriving diagnostic labels.
        """

        logger.info("Loading PTB-XL metadata files...")

        main_csv_path = self.dataset_path / "ptbxl_database.csv"
        if not main_csv_path.exists():  # pragma: no cover - depends on dataset
            raise FileNotFoundError(f"PTB-XL database file not found: {main_csv_path}")

        main_df = pd.read_csv(main_csv_path, index_col="ecg_id")
        logger.info("Loaded PTB-XL database with %d records", len(main_df))

        scp_csv_path = self.dataset_path / "scp_statements.csv"
        if not scp_csv_path.exists():  # pragma: no cover - depends on dataset
            raise FileNotFoundError(
                f"SCP statements file not found: {scp_csv_path}"
            )

        scp_df = pd.read_csv(scp_csv_path, index_col=0)
        logger.info("Loaded SCP statements with %d entries", len(scp_df))

        return main_df, scp_df

    # ------------------------------------------------------------------
    # ECG signal loading
    # ------------------------------------------------------------------
    def load_ecg_signals(
        self,
        df: pd.DataFrame,
        sampling_rate: int,
    ) -> np.ndarray:
        """Load ECG signals using ``wfdb``.

        Parameters
        ----------
        df:
            DataFrame containing the ``filename_hr`` and ``filename_lr`` columns.
        sampling_rate:
            Desired sampling rate for the signals. Must be present in
            ``config.sampling_rates``.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_samples, n_timesteps, n_leads)``.
        """

        logger.info("Loading ECG signals at %dHz", sampling_rate)

        if sampling_rate == self.config.sampling_rates["hr"]:
            filename_col = "filename_hr"
        elif sampling_rate == self.config.sampling_rates["lr"]:
            filename_col = "filename_lr"
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported sampling rate: {sampling_rate}")

        if filename_col not in df.columns:
            raise ValueError(f"Column '{filename_col}' not found in dataframe")

        signals: List[np.ndarray] = []
        failed = 0
        start_time = time.time()

        for idx, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Loading ECG", dynamic_ncols=True),
            start=1,
        ):
            filename = str(row[filename_col]).lstrip("/")
            signal_path = self.dataset_path / filename

            dat_path = signal_path.with_suffix(".dat")
            hea_path = signal_path.with_suffix(".hea")
            if not dat_path.exists() or not hea_path.exists():
                logger.warning("Missing WFDB files for %s", signal_path)
                failed += 1
                continue

            try:
                signal, _ = wfdb.rdsamp(str(signal_path))
            except Exception as exc:  # pragma: no cover - depends on dataset
                logger.error("Failed to load %s: %s", signal_path, exc)
                failed += 1
                continue

            if signal.size == 0:
                logger.warning("Empty ECG signal for record %s", signal_path)
                failed += 1
                continue

            expected_samples = self.config.expected_duration * sampling_rate
            if signal.shape[0] != expected_samples:
                logger.warning(
                    "Record %s has unexpected length %s (expected %s)",
                    signal_path,
                    signal.shape[0],
                    expected_samples,
                )
            if signal.shape[1] != self.config.expected_leads:
                logger.warning(
                    "Record %s has %s leads (expected %s)",
                    signal_path,
                    signal.shape[1],
                    self.config.expected_leads,
                )

            signals.append(signal)

            if idx % 500 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                rate = idx / elapsed
                remaining = (len(df) - idx) / max(rate, 1e-6)
                logger.info(
                    "Loaded %d/%d records | %.2f rec/s | ETA %.1f min",
                    idx,
                    len(df),
                    rate,
                    remaining / 60,
                )

        if not signals:
            raise ValueError("No ECG signals could be loaded")

        logger.info(
            "Successfully loaded %d signals (failed: %d)", len(signals), failed
        )
        return np.asarray(signals)

    # ------------------------------------------------------------------
    def debug_paths(self, df: pd.DataFrame, sampling_rate: int = 500, n: int = 5) -> None:
        """Log a subset of WFDB paths for debugging purposes."""

        if sampling_rate == self.config.sampling_rates["hr"]:
            filename_col = "filename_hr"
        else:
            filename_col = "filename_lr"

        logger.info("Inspecting %d WFDB paths", n)
        for _, row in df.head(n).iterrows():
            filename = str(row[filename_col]).lstrip("/")
            base = self.dataset_path / filename
            logger.info(
                "%s | .dat: %s | .hea: %s",
                base,
                base.with_suffix(".dat").exists(),
                base.with_suffix(".hea").exists(),
            )


class PTBXLPreprocessor:
    """Collection of preprocessing routines for PTB-XL metadata."""

    def __init__(self, config: PTBXLConfig) -> None:
        self.config = config
        self.scaler = self._get_scaler()
        self.is_fitted = False

        np.random.seed(config.random_seed)

    # ------------------------------------------------------------------
    def _get_scaler(self):
        scalers = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        return scalers.get(self.config.scaler_type, RobustScaler())

    # ------------------------------------------------------------------
    def preprocess_age(self, ages: pd.Series) -> pd.Series:
        """Decode and scale the age column according to PTB-XL rules."""

        logger.info("Scaling patient ages")
        ages_corrected = ages.copy()

        encoded_mask = ages > self.config.age_encoding_threshold
        ages_corrected.loc[encoded_mask] = ages.loc[encoded_mask] - 300
        logger.info("Corrected %d encoded age values", int(encoded_mask.sum()))

        min_age, max_age = self.config.age_bounds
        outliers = (ages_corrected < min_age) | (ages_corrected > max_age)
        if outliers.any():
            replacement = ages_corrected.loc[~outliers].median()
            logger.warning("Replacing %d age outliers with %.2f", int(outliers.sum()), replacement)
            ages_corrected.loc[outliers] = replacement

        values = ages_corrected.values.reshape(-1, 1)
        if not self.is_fitted:
            scaled = self.scaler.fit_transform(values)
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(values)

        return pd.Series(scaled.flatten(), index=ages.index, name="age")

    # ------------------------------------------------------------------
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Perform one-hot encoding for categorical metadata columns."""

        df_encoded = df.copy()

        if "sex" in df_encoded.columns:
            sex_encoded = pd.get_dummies(df_encoded["sex"], prefix="sex", dtype=int)
            rename_map = {"sex_0": "sex_Female", "sex_1": "sex_Male"}
            sex_encoded = sex_encoded.rename(columns=rename_map)
            for column in ["sex_Female", "sex_Male"]:
                if column not in sex_encoded.columns:
                    sex_encoded[column] = 0
            df_encoded = df_encoded.drop(columns=["sex"])
            df_encoded = pd.concat(
                [df_encoded, sex_encoded[["sex_Female", "sex_Male"]]], axis=1
            )
            logger.info("Encoded sex column using one-hot representation")
        else:
            for column in ["sex_Female", "sex_Male"]:
                if column not in df_encoded.columns:
                    df_encoded[column] = 0

        return df_encoded

    # ------------------------------------------------------------------
    def create_diagnostic_labels(
        self, df: pd.DataFrame, scp_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create multi-label diagnostic labels from SCP statements."""

        logger.info("Generating diagnostic labels")

        if "diagnostic" in scp_df.columns:
            diagnostic_codes = scp_df[scp_df["diagnostic"] == 1]
        else:
            diagnostic_codes = scp_df

        def parse_codes(raw_codes: Any) -> Dict[str, float]:
            if isinstance(raw_codes, str):
                try:
                    return ast.literal_eval(raw_codes)
                except (SyntaxError, ValueError):  # pragma: no cover - defensive
                    return {}
            return raw_codes if isinstance(raw_codes, dict) else {}

        parsed_codes = df["scp_codes"].apply(parse_codes)

        def map_to_superclass(codes: Dict[str, float]) -> Dict[str, int]:
            labels = {cls: 0 for cls in self.config.diagnostic_classes}
            for code in codes.keys():
                if code in diagnostic_codes.index:
                    superclass = diagnostic_codes.loc[code, "diagnostic_class"]
                    if superclass in labels:
                        labels[superclass] = 1
            return labels

        diagnostic_labels = parsed_codes.apply(map_to_superclass)
        diagnostic_df = pd.DataFrame(diagnostic_labels.tolist(), index=df.index)

        label_counts = diagnostic_df.sum()
        total = len(diagnostic_df)
        for label in diagnostic_df.columns:
            logger.info(
                "Label %-5s -> %6d samples (%.1f%%)",
                label,
                label_counts[label],
                100 * label_counts[label] / max(total, 1),
            )

        missing_labels = int((diagnostic_df.sum(axis=1) == 0).sum())
        if missing_labels:
            logger.warning("%d samples do not contain diagnostic labels", missing_labels)

        return diagnostic_df

    # ------------------------------------------------------------------
    def create_official_splits(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate train/val/test splits using PTB-XL stratified folds."""

        if "strat_fold" not in df.columns:
            raise ValueError("The dataframe is missing the 'strat_fold' column")

        train_indices = df[df["strat_fold"].isin(self.config.train_folds)].index.values
        val_indices = df[df["strat_fold"] == self.config.val_fold].index.values
        test_indices = df[df["strat_fold"] == self.config.test_fold].index.values

        logger.info(
            "Using official PTB-XL folds -> train: %d | val: %d | test: %d",
            len(train_indices),
            len(val_indices),
            len(test_indices),
        )

        return {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def create_splits_from_indices(
        X_ecg: np.ndarray,
        X_features: pd.DataFrame,
        y: pd.DataFrame,
        split_indices: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Slice arrays/DataFrames according to stratified fold indices."""

        def index_positions(indices: np.ndarray) -> List[int]:
            return [X_features.index.get_loc(idx) for idx in indices]

        train_pos = index_positions(split_indices["train_indices"])
        val_pos = index_positions(split_indices["val_indices"])
        test_pos = index_positions(split_indices["test_indices"])

        return {
            "X_ecg_train": X_ecg[train_pos],
            "X_ecg_val": X_ecg[val_pos],
            "X_ecg_test": X_ecg[test_pos],
            "X_features_train": X_features.iloc[train_pos],
            "X_features_val": X_features.iloc[val_pos],
            "X_features_test": X_features.iloc[test_pos],
            "y_train": y.iloc[train_pos],
            "y_val": y.iloc[val_pos],
            "y_test": y.iloc[test_pos],
            "train_indices": train_pos,
            "val_indices": val_pos,
            "test_indices": test_pos,
        }


class PTBXLPipeline:
    """High level pipeline orchestrating the preprocessing steps."""

    def __init__(self, config: PTBXLConfig) -> None:
        self.config = config
        self.loader = PTBXLDataLoader(config)
        self.preprocessor = PTBXLPreprocessor(config)

    # ------------------------------------------------------------------
    def run_complete_preprocessing(
        self,
        sampling_rate: Optional[int] = None,
        max_samples: Optional[int] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete preprocessing workflow."""

        logger.info("%s", "=" * 80)
        logger.info("Starting PTB-XL preprocessing pipeline")
        logger.info("%s", "=" * 80)

        sampling_rate = sampling_rate or self.config.default_sampling_rate
        if sampling_rate not in self.config.sampling_rates.values():
            raise ValueError(
                "Sampling rate %s is not supported. Expected one of %s"
                % (sampling_rate, list(self.config.sampling_rates.values()))
            )

        main_df, scp_df = self.loader.load_metadata()
        sample_df = main_df.copy()
        if max_samples is not None:
            sample_df = sample_df.head(max_samples)
            logger.info("Processing subset with %d samples", len(sample_df))

        X_ecg = self.loader.load_ecg_signals(sample_df, sampling_rate=sampling_rate)

        sample_df = sample_df.copy()
        sample_df["age"] = self.preprocessor.preprocess_age(sample_df["age"])
        sample_df = self.preprocessor.encode_categorical_features(sample_df)

        feature_columns = ["age", "sex_Female", "sex_Male"]
        for column in feature_columns:
            if column not in sample_df.columns:
                sample_df[column] = 0
        X_features = sample_df[feature_columns].copy()

        y = self.preprocessor.create_diagnostic_labels(sample_df, scp_df)

        if self.config.use_official_folds:
            split_indices = self.preprocessor.create_official_splits(sample_df)
            splits = self.preprocessor.create_splits_from_indices(
                X_ecg, X_features, y, split_indices
            )
        else:  # pragma: no cover - alternative splitting strategy not implemented
            raise NotImplementedError(
                "Random data splits are not implemented. Set 'use_official_folds' to True."
            )

        dataset: Dict[str, Any] = {
            "X_ecg": X_ecg,
            "X_features": X_features,
            "y": y,
            "dataframe": sample_df,
            "splits": splits,
            "feature_names": feature_columns,
            "label_names": list(y.columns),
            "config": self.config,
            "preprocessing_info": {
                "total_samples": int(len(sample_df)),
                "ecg_shape": tuple(X_ecg.shape),
                "n_features": len(feature_columns),
                "n_labels": len(y.columns),
                "sampling_rate": sampling_rate,
                "train_samples": int(len(splits["X_ecg_train"])),
                "val_samples": int(len(splits["X_ecg_val"])),
                "test_samples": int(len(splits["X_ecg_test"])),
            },
        }

        if save:
            self.save_dataset(dataset)

        logger.info("%s", "=" * 80)
        logger.info("PTB-XL preprocessing completed successfully")
        logger.info("%s", "=" * 80)

        return dataset

    # ------------------------------------------------------------------
    def save_dataset(self, dataset: Dict[str, Any]) -> None:
        """Persist the processed dataset to disk."""

        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving preprocessed dataset to %s", output_dir)

        splits = dataset["splits"]
        np.save(output_dir / "X_ecg_train.npy", splits["X_ecg_train"])
        np.save(output_dir / "X_ecg_val.npy", splits["X_ecg_val"])
        np.save(output_dir / "X_ecg_test.npy", splits["X_ecg_test"])

        splits["X_features_train"].to_csv(output_dir / "X_features_train.csv")
        splits["X_features_val"].to_csv(output_dir / "X_features_val.csv")
        splits["X_features_test"].to_csv(output_dir / "X_features_test.csv")

        splits["y_train"].to_csv(output_dir / "y_train.csv")
        splits["y_val"].to_csv(output_dir / "y_val.csv")
        splits["y_test"].to_csv(output_dir / "y_test.csv")

        metadata = {
            "feature_names": dataset["feature_names"],
            "label_names": dataset["label_names"],
            "preprocessing_info": dataset["preprocessing_info"],
        }

        with open(output_dir / "dataset_metadata.pkl", "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)

        with open(output_dir / "config.pkl", "wb") as config_file:
            pickle.dump(dataset["config"], config_file)

        info = dataset["preprocessing_info"]
        summary = (
            f"Total samples: {info['total_samples']:,}\n"
            f"ECG shape: {info['ecg_shape']}\n"
            f"Features ({info['n_features']}): {', '.join(dataset['feature_names'])}\n"
            f"Labels ({info['n_labels']}): {', '.join(dataset['label_names'])}\n"
            f"Train/Val/Test: {info['train_samples']:,}/"
            f"{info['val_samples']:,}/{info['test_samples']:,}"
        )
        logger.info("Preprocessing summary:\n%s", summary)


def create_visualizations(dataset: Dict[str, Any], output_file: Optional[Path] = None) -> None:
    """Generate high level visualisations describing the processed dataset."""

    import matplotlib.pyplot as plt  # Imported lazily to avoid heavy dependency
    import seaborn as sns  # type: ignore

    splits = dataset["splits"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PTB-XL Preprocessing Results", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    split_sizes = [
        len(splits["X_ecg_train"]),
        len(splits["X_ecg_val"]),
        len(splits["X_ecg_test"]),
    ]
    ax1.pie(split_sizes, labels=["Train", "Validation", "Test"], autopct="%1.1f%%")
    ax1.set_title("Data Split Distribution")

    ax2 = axes[0, 1]
    y_train = splits["y_train"]
    label_counts = y_train.sum()
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax2, color="skyblue")
    ax2.set_title("Training Set Label Distribution")
    ax2.set_xlabel("Diagnostic Classes")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)
    for bar in ax2.patches:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
        )

    ax3 = axes[0, 2]
    X_features_train = splits["X_features_train"]
    ax3.hist(X_features_train["age"], bins=30, alpha=0.7, color="green")
    ax3.set_title("Age Distribution (scaled)")
    ax3.set_xlabel("Age (scaled)")

    ax4 = axes[1, 0]
    if len(splits["X_ecg_train"]) > 0:
        sample_ecg = splits["X_ecg_train"][0]
        time_axis = np.arange(sample_ecg.shape[0]) / dataset["preprocessing_info"]["sampling_rate"]
        ax4.plot(time_axis, sample_ecg[:, 0], linewidth=0.8, color="red")
        ax4.set_title("Sample ECG Signal (Lead I)")
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Amplitude")
        ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    sex_counts = [
        (X_features_train["sex_Female"] == 1).sum(),
        (X_features_train["sex_Male"] == 1).sum(),
    ]
    ax5.bar(["Female", "Male"], sex_counts, color=["pink", "lightblue"], alpha=0.7)
    ax5.set_title("Sex Distribution (Train)")
    for idx, count in enumerate(sex_counts):
        ax5.text(idx, count, f"{count}", ha="center", va="bottom")

    ax6 = axes[1, 2]
    labels_per_sample = y_train.sum(axis=1)
    label_dist = labels_per_sample.value_counts().sort_index()
    ax6.bar(label_dist.index, label_dist.values, alpha=0.7, color="orange")
    ax6.set_title("Labels per Sample Distribution")
    ax6.set_xlabel("Number of Labels")
    ax6.set_ylabel("Samples")

    plt.tight_layout()
    if output_file is None:
        output_file = Path("ptbxl_preprocessing_results.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Visualisations saved to %s", output_file)


# ---------------------------------------------------------------------------
# Backwards compatible helper
# ---------------------------------------------------------------------------
def preprocess_data(
    data_path: str,
    scp_statements_path: str,
    output_path: Optional[str],
    sampling_rate: int,
    path: Optional[str] = None,
    *,
    max_samples: Optional[int] = None,
    scaler_type: str = "robust",
    use_official_folds: bool = True,
    save_intermediate: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess PTB-XL data while maintaining the legacy API.

    Parameters
    ----------
    data_path:
        Path to ``ptbxl_database.csv``.
    scp_statements_path:
        Path to ``scp_statements.csv``.
    output_path:
        Optional file where a compact CSV (features + labels) will be stored.
    sampling_rate:
        Desired sampling rate (typically 500 or 100).
    path:
        Root directory of the PTB-XL dataset. When ``None`` it defaults to the
        parent directory of ``data_path``.
    max_samples:
        If provided, restricts the preprocessing to the first ``n`` samples.
    scaler_type:
        Which scaler to use for the age column (``robust``, ``standard`` or
        ``minmax``).
    use_official_folds:
        When ``True`` the official PTB-XL stratified folds are used for
        splitting. Any other value raises ``NotImplementedError`` to avoid
        silent fallbacks.
    save_intermediate:
        When ``True`` the pipeline also saves the split arrays/frames in the
        directory configured via :class:`PTBXLConfig`.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``X_ecg``, ``X_features``, ``data`` (metadata dataframe) and the
        diagnostic label dataframe ``Y``.
    """

    root_path = Path(path) if path is not None else Path(data_path).resolve().parent

    config = PTBXLConfig(
        dataset_path=root_path,
        output_path=Path(output_path).resolve().parent if output_path else root_path / "preprocessed",
        scaler_type=scaler_type,
        use_official_folds=use_official_folds,
    )

    pipeline = PTBXLPipeline(config)
    dataset = pipeline.run_complete_preprocessing(
        sampling_rate=sampling_rate,
        max_samples=max_samples,
        save=save_intermediate,
    )

    X_ecg = dataset["X_ecg"]
    X_features = dataset["X_features"]
    data = dataset["dataframe"]
    Y = dataset["y"]

    if output_path:
        combined = pd.concat([X_features, Y], axis=1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path)
        logger.info("Saved combined feature/label CSV to %s", output_path)

    return X_ecg, X_features, data, Y


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL preprocessing pipeline")
    parser.add_argument("dataset", type=Path, help="Path to PTB-XL dataset directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where processed arrays/frames will be stored",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV file combining features and labels",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=500,
        help="Sampling rate to use for loading ECG signals",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit preprocessing to the first N samples",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard", "minmax"],
        help="Scaler to use for the age column",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving numpy/csv artefacts and only run in-memory preprocessing",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or (args.dataset / "preprocessed")

    config = PTBXLConfig(
        dataset_path=args.dataset,
        output_path=output_dir,
        scaler_type=args.scaler,
    )

    pipeline = PTBXLPipeline(config)
    dataset = pipeline.run_complete_preprocessing(
        sampling_rate=args.sampling_rate,
        max_samples=args.max_samples,
        save=not args.no_save,
    )

    if args.output_csv:
        combined = pd.concat([dataset["X_features"], dataset["y"]], axis=1)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_csv)
        logger.info("Saved combined dataset CSV to %s", args.output_csv)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
