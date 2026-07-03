from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Colab-ready dataset zip with real files, not symlinks.")
    parser.add_argument(
        "--dataset",
        default="datasets/mixed_indoor_fire_risk_plus_new",
        help="Dataset directory to package.",
    )
    parser.add_argument(
        "--output",
        default="fire_detection_datasets.zip",
        help="Output zip path. Upload this file to Google Drive for the Colab notebook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)

    if not (dataset / "data.yaml").exists():
        raise SystemExit(f"Missing dataset yaml: {dataset / 'data.yaml'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        staged_dataset = tmp_root / "datasets" / dataset.name
        staged_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset, staged_dataset, symlinks=False)

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for path in sorted((tmp_root / "datasets").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp_root))

    print(f"wrote {output.resolve()}")
    print("Upload this zip to Google Drive and set DATA_ZIP in notebooks/fire_risk_colab_training.ipynb.")


if __name__ == "__main__":
    main()
