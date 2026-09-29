#!/usr/bin/env python3
"""Validate the generated COCO-PARTIAL 10% split against train2017."""

import argparse
import hashlib
import json
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique(values, description):
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {description}: {len(values) - len(set(values))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path, help="COCO root")
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    annotations = args.data_root / "annotations"
    full_path = annotations / "instances_train2017.json"
    split_dir = annotations / "semi_supervised"
    labeled_path = split_dir / f"instances_train2017.{args.fold}@{args.percent}.json"
    unlabeled_path = split_dir / f"instances_train2017.{args.fold}@{args.percent}-unlabeled.json"
    full, labeled, unlabeled = map(load_json, (full_path, labeled_path, unlabeled_path))

    full_ids = [image["id"] for image in full["images"]]
    labeled_ids = [image["id"] for image in labeled["images"]]
    unlabeled_ids = [image["id"] for image in unlabeled["images"]]
    require_unique(full_ids, "train2017 image IDs")
    require_unique(labeled_ids, "labeled image IDs")
    require_unique(unlabeled_ids, "unlabeled image IDs")

    full_set, labeled_set, unlabeled_set = set(full_ids), set(labeled_ids), set(unlabeled_ids)
    overlap = labeled_set & unlabeled_set
    missing = full_set - (labeled_set | unlabeled_set)
    unexpected = (labeled_set | unlabeled_set) - full_set
    if overlap or missing or unexpected:
        raise ValueError(
            f"invalid split: overlap={len(overlap)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    expected_labeled = int(args.percent / 100.0 * len(full_ids))
    if len(labeled_ids) != expected_labeled:
        raise ValueError(f"expected {expected_labeled} labeled images, got {len(labeled_ids)}")

    for name, dataset in (("full", full), ("labeled", labeled), ("unlabeled", unlabeled)):
        image_ids = {image["id"] for image in dataset["images"]}
        annotation_image_ids = [annotation["image_id"] for annotation in dataset["annotations"]]
        invalid = set(annotation_image_ids) - image_ids
        if invalid:
            raise ValueError(f"{name} annotations reference {len(invalid)} absent image IDs")
        category_ids = {category["id"] for category in dataset["categories"]}
        invalid_categories = {
            annotation["category_id"]
            for annotation in dataset["annotations"]
            if annotation["category_id"] not in category_ids
        }
        if invalid_categories:
            raise ValueError(f"{name} annotations reference invalid category IDs")
        if category_ids != {category["id"] for category in full["categories"]}:
            raise ValueError(f"{name} category IDs differ from the full train2017 set")
        require_unique([annotation["id"] for annotation in dataset["annotations"]], f"{name} annotation IDs")

    full_annotation_ids = {annotation["id"] for annotation in full["annotations"]}
    split_annotation_ids = {
        annotation["id"]
        for dataset in (labeled, unlabeled)
        for annotation in dataset["annotations"]
    }
    if full_annotation_ids != split_annotation_ids:
        raise ValueError(
            f"annotation union mismatch: missing={len(full_annotation_ids - split_annotation_ids)}, "
            f"unexpected={len(split_annotation_ids - full_annotation_ids)}"
        )

    if args.check_files:
        image_dir = args.data_root / "train2017"
        missing_files = [
            image["file_name"]
            for image in full["images"]
            if not (image_dir / image["file_name"]).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} train2017 image files are missing")

    print(json.dumps({
        "split_provenance": "generated with tools/dataset/semi_coco.py; not asserted to be the authors' original split",
        "fold_seed": args.fold,
        "percent": args.percent,
        "train2017_images": len(full_ids),
        "labeled_images": len(labeled_ids),
        "unlabeled_images": len(unlabeled_ids),
        "intersection": len(overlap),
        "union_matches_train2017": True,
        "train2017_annotations": len(full["annotations"]),
        "labeled_annotations": len(labeled["annotations"]),
        "unlabeled_annotations": len(unlabeled["annotations"]),
        "labeled_json_sha256": sha256(labeled_path),
        "unlabeled_json_sha256": sha256(unlabeled_path),
    }, indent=2))


if __name__ == "__main__":
    main()
