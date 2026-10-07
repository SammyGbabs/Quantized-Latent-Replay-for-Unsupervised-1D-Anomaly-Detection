# preprocess.py
from pathlib import Path
import argparse

import pandas as pd
import torch


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

RAW_SKAB_DIR = Path("data/raw_data/skab")
OUTPUT_SKAB_DIR = Path("preprocessed/skab")


# -------------------------------------------------------------------
# SKAB preprocessing
# -------------------------------------------------------------------

def preprocess_skab_file(csv_path: Path, output_path: Path):
    """
    Convert one SKAB CSV file into a PyTorch .pt file.

    Output data shape:
        (n_sensors, n_timesteps)

    The saved dictionary contains:
        - data: sensor tensor
        - anomaly: anomaly labels, if available
        - changepoint: changepoint labels, if available
        - sensor_names: list of sensor names
        - source_file: original CSV path
    """

    print(f"Processing: {csv_path}")

    # SKAB uses semicolon-separated CSV files.
    df = pd.read_csv(csv_path, sep=";")

    # Make sure all expected sensor columns are present.
    missing_sensors = [
        col for col in SENSOR_COLUMNS
        if col not in df.columns
    ]

    if missing_sensors:
        raise ValueError(
            f"Missing sensor columns in {csv_path}: "
            f"{missing_sensors}"
        )

    # Extract sensor measurements.
    #
    # pandas gives us:
    #     (timesteps, sensors)
    #
    # We transpose it because the model expects:
    #     (channels, length)
    sensor_data = df[SENSOR_COLUMNS].to_numpy()

    data = torch.tensor(
        sensor_data,
        dtype=torch.float32
    ).T

    # Optional ground-truth labels.
    anomaly = None
    changepoint = None

    if "anomaly" in df.columns:
        anomaly = torch.tensor(
            df["anomaly"].to_numpy(),
            dtype=torch.long
        )

    if "changepoint" in df.columns:
        changepoint = torch.tensor(
            df["changepoint"].to_numpy(),
            dtype=torch.long
        )

    # Store everything together.
    processed = {
        "data": data,
        "anomaly": anomaly,
        "changepoint": changepoint,
        "sensor_names": SENSOR_COLUMNS,
        "source_file": str(csv_path),
    }

    # Create destination directory.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(processed, output_path)

    print(
        f"  Saved: {output_path} "
        f"| shape={tuple(data.shape)}"
    )


def preprocess_skab(
    raw_dir: Path = RAW_SKAB_DIR,
    output_dir: Path = OUTPUT_SKAB_DIR,
    overwrite: bool = False,
):
    """
    Preprocess every SKAB CSV file recursively.
    """

    csv_files = sorted(raw_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}"
        )

    print(f"Found {len(csv_files)} SKAB CSV files.\n")

    processed_count = 0
    skipped_count = 0

    for csv_path in csv_files:

        # Preserve the directory structure.
        #
        # Example:
        # raw_data/skab/valve1/3.csv
        #
        # becomes:
        # preprocessed/skab/valve1/3.pt
        relative_path = csv_path.relative_to(raw_dir)
        output_path = (
            output_dir
            / relative_path.with_suffix(".pt")
        )

        if output_path.exists() and not overwrite:
            print(f"Skipping existing file: {output_path}")
            skipped_count += 1
            continue

        preprocess_skab_file(
            csv_path,
            output_path
        )

        processed_count += 1

    print("\n" + "=" * 60)
    print("SKAB preprocessing complete")
    print("=" * 60)
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Output:    {output_dir}")


# -------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SKAB into PyTorch tensors."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files that already exist."
    )

    args = parser.parse_args()

    preprocess_skab(
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
