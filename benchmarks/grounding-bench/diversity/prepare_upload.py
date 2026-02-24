import argparse
import json
import os
import shutil
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

LABEL_MAP = {0: "TextBox", 1: "ChoiceButton", 2: "Signature"}


def convert_bbox_to_ls(
    bbox: list[float], img_w: int, img_h: int, label_id: int
) -> dict:
    """
    Converts a bounding box from COCO format [x, y, w, h] to Label Studio percentage format.
    """
    x, y, w, h = bbox

    label_str = LABEL_MAP.get(label_id, str(label_id))

    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": (x / img_w) * 100,
            "y": (y / img_h) * 100,
            "width": (w / img_w) * 100,
            "height": (h / img_h) * 100,
            "rotation": 0,
            "rectanglelabels": [label_str],
        },
    }


def load_existing_exclusions(jsonl_path: str) -> set[str]:
    """
    Parses a Label Studio JSONL file to find filenames that are already annotated.
    Returns a set of basenames (e.g., 'image.png').
    """
    excluded = set()
    if not jsonl_path:
        return excluded

    path = Path(jsonl_path)
    if not path.exists():
        print(f"Warning: Exclusion file {jsonl_path} not found.")
        return excluded

    print(f"Loading exclusions from {jsonl_path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                # Handle different Label Studio export formats
                # 1. Direct task format: {"data": {"image": "..."}}
                # 2. Export format: {"task": {"data": {"image": "..."}}}
                data = task.get("data", {})
                if "image" not in data and "task" in task:
                    data = task["task"].get("data", {})

                image_url = data.get("image", "")

                if image_url:
                    # Parse url to get filename
                    parsed = urllib.parse.urlparse(image_url)
                    filename = os.path.basename(parsed.path)
                    excluded.add(filename)
            except json.JSONDecodeError:
                pass

    print(f"Found {len(excluded)} existing files/tasks to ignore.")
    return excluded


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images and Label Studio tasks from ranked data."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to input Parquet file OR directory of images"
    )
    parser.add_argument("ranked_csv", type=str, help="Path to visual ranking CSV")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save output images and tasks"
    )
    parser.add_argument(
        "--top-k", type=int, default=175, help="Number of top-ranked images to select"
    )
    parser.add_argument(
        "--blob-prefix",
        type=str,
        default="azure-blob://your-container/your-path",
        help="Prefix for Azure Blob Storage URLs (e.g., azure-blob://container/path)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Optional path to an existing JSONL file. Images in this file will be skipped.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input path not found: {args.input_path}")
        sys.exit(1)

    # 1. Load Exclusions
    excluded_basenames = load_existing_exclusions(args.exclude)

    # 2. Retrieve top-ranked filenames
    print(f"Reading rankings from {args.ranked_csv}...")
    try:
        df_rank = pd.read_csv(args.ranked_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if "rank" not in df_rank.columns:
        print("Error: CSV must contain a 'rank' column.")
        sys.exit(1)

    df_rank = df_rank.sort_values("rank")

    # Select files that are not excluded
    selected_files = []

    # df_rank['filename'] usually contains the relative path used during ranking.
    # We need to be careful about matching.
    # Usually output of this script flattens images locally (os.path.basename).
    # Task JSONs also usually point to basenames in the bucket.
    # So we compare basenames.

    for filename in df_rank["filename"].values:
        basename = os.path.basename(filename)
        if basename not in excluded_basenames:
            selected_files.append(filename)

        if len(selected_files) >= args.top_k:
            break

    if not selected_files:
        print(
            "No files selected (all top ranked files might be excluded or limit is 0)."
        )
        sys.exit(0)

    print(f"Targeting top {len(selected_files)} images (from available rankings).")

    # Set needed for fast lookup later
    # Note: selected_files contains the identifiers from the ranking CSV (could be relative paths)
    target_files_set = set(selected_files)

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = []

    # 3. Process Input
    if os.path.isdir(args.input_path):
        print(f"Processing directory: {args.input_path}")

        for filename in tqdm(selected_files, desc="Copying images"):
            # Assume filename in ranking is relative to input_path
            # (or absolute, but rank.py usually outputs relative if given a dir)
            src_path = os.path.join(args.input_path, filename)

            if not os.path.exists(src_path):
                # Try checking if rank filename was absolute or relative differently
                # But for now assume standard rank.py behavior
                print(f"Warning: File {src_path} not found. Skipping.")
                continue

            clean_name = os.path.basename(filename)
            dst_path = os.path.join(args.output_dir, clean_name)

            shutil.copy2(src_path, dst_path)

            # Construct Task (no predictions for directory mode yet)
            prefix = args.blob_prefix.rstrip("/")
            task = {
                "data": {"image": f"{prefix}/{clean_name}"},
                "predictions": [],
            }
            tasks.append(task)

            # Write individual JSON
            task_file = os.path.join(
                args.output_dir, f"{os.path.splitext(clean_name)[0]}.json"
            )
            with open(task_file, "w") as f:
                json.dump(task, f, indent=2)

    else:
        # Parquet Logic
        print(f"Reading parquet {args.input_path}...")
        df = pd.read_parquet(args.input_path)

        # Identify filename column
        files_col = None
        for col in ["file_name", "filename"]:
            if col in df.columns:
                files_col = col
                break

        if not files_col:
            print("Error: Parquet file must have 'file_name' or 'filename' column.")
            sys.exit(1)

        # Filter DF to only selected files
        df_subset = df[df[files_col].isin(target_files_set)].copy()
        print(f"Found {len(df_subset)} matching images in parquet.")

        for idx, row in tqdm(
            df_subset.iterrows(), total=len(df_subset), desc="Extracting images"
        ):
            filename = row[files_col]

            # Standardize image data format
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
            else:
                img_bytes = img_data

            width = row["width"]
            height = row["height"]

            # Save Image
            clean_name = os.path.basename(filename)
            img_path = os.path.join(args.output_dir, clean_name)

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Process Pre-annotations
            objects = row.get("objects", {})
            predictions = []

            # Handle object format from different parquet schemas if needed,
            # here assuming standard dataset dict format
            if objects and "bbox" in objects:
                bboxes = objects["bbox"]
                categories = objects["category"]

                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    cat = categories[i]

                    ls_region = convert_bbox_to_ls(bbox, width, height, cat)
                    predictions.append(ls_region)

            # Construct Label Studio Task
            prefix = args.blob_prefix.rstrip("/")
            task = {
                "data": {"image": f"{prefix}/{clean_name}"},
                "predictions": [
                    {"model_version": "pre-annotation", "result": predictions}
                ],
            }

            tasks.append(task)

            # Write individual JSON task file
            task_file = os.path.join(
                args.output_dir, f"{os.path.splitext(clean_name)[0]}.json"
            )
            with open(task_file, "w") as f:
                json.dump(task, f, indent=2)

    # Write Master JSONL
    output_name = os.path.basename(os.path.normpath(args.output_dir))
    jsonl_path = os.path.join(args.output_dir, f"{output_name}-tasks.jsonl")
    with open(jsonl_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Done. Output in {args.output_dir}/")
    print(f"Created {len(tasks)} images and task files.")
    print(f"Master task list: {jsonl_path}")


if __name__ == "__main__":
    main()
