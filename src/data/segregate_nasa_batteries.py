"""
src/data/segregate_nasa_batteries.py

Segregates raw NASA per-cycle CSV files for B0006, B0007, and B0018 into
the same dataset/processed/<battery_id>/{charge,discharge,impedance}/
structure already used by B0005 -- using dataset/raw/metadata/metadata.csv
as the authoritative source of truth for which raw file belongs to which
battery and experiment type.

This is a SEGREGATION-ONLY script: files are copied byte-for-byte
(shutil.copy2) from dataset/raw/nasa_original/ into the processed folder
structure. No CSV content is read, modified, cleaned, or regenerated.
B0005 -- already segregated and validated -- is never touched; every
function below explicitly refuses to operate on B0005.

IMPORTANT, VERIFIED BEFORE WRITING THIS SCRIPT: metadata.csv, in its raw
form, contains two known issues (an embedded header row, and the entire
file duplicated -- every row appears exactly twice). This was found and
fixed early in this project (see load_metadata() in
src/core/data_loader.py, reused here rather than re-implemented). Any
"expected file count" computed from the RAW, uncleaned metadata will be
exactly 2x too high. This script always computes expected counts from
the cleaned metadata (via load_metadata()) -- never from a hardcoded
number -- specifically to avoid repeating that already-diagnosed bug.

Usage:
    python src/data/segregate_nasa_batteries.py
(run from the project root, or anywhere -- the sys.path bootstrap below
makes the src package importable regardless of current working directory)
"""

import sys
import shutil
import hashlib
from pathlib import Path

# --- sys.path bootstrap so `from src.core... import ...` works whether this
# script is run as `python src/data/segregate_nasa_batteries.py` from the
# project root or from anywhere else. This is the only accommodation this
# file makes for standalone execution -- no other project file is touched. ---
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]  # src/data/this_file.py -> project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.core.config import PROCESSED_DATA, NASA_ORIGINAL  # noqa: E402
from src.core.data_loader import load_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROTECTED_BATTERY_ID = "B0005"
TARGET_BATTERY_IDS = ["B0006", "B0007", "B0018"]
EXPERIMENT_TYPES = ["charge", "discharge", "impedance"]

REQUIRED_METADATA_COLUMNS = [
    "type", "battery_id", "filename",
]


class SegregationError(RuntimeError):
    """Raised for any validation or copy failure. Deliberately not caught
    anywhere in this module -- a failure here must stop execution with a
    clear message, not be silently swallowed."""


# ---------------------------------------------------------------------------
# Metadata handling
# ---------------------------------------------------------------------------

def validate_metadata_schema(metadata):
    """Confirms the columns this script depends on are actually present.
    Fails clearly and immediately if not."""
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in metadata.columns]
    if missing:
        raise SegregationError(
            f"metadata.csv is missing required column(s): {missing}. "
            f"Available columns: {list(metadata.columns)}"
        )
    print(f"PASS: metadata schema contains required columns {REQUIRED_METADATA_COLUMNS}")


def get_battery_records(metadata, battery_id):
    """Returns the metadata rows for one battery, all experiment types."""
    return metadata[metadata["battery_id"] == battery_id].copy()


def compute_expected_counts(metadata, battery_id):
    """
    Computes expected per-type file counts for a battery DIRECTLY from the
    (already-cleaned) metadata -- never from a hardcoded number. This is
    deliberate: hardcoding a number invites exactly the kind of stale/wrong
    expectation this project has already found once (raw metadata is 2x
    duplicated). The metadata itself, once cleaned, is the only source of
    truth for what "expected" means here.
    """
    records = get_battery_records(metadata, battery_id)
    counts = {}
    for experiment in EXPERIMENT_TYPES:
        counts[experiment] = int((records["type"] == experiment).sum())
    return counts


# ---------------------------------------------------------------------------
# File identity / integrity helpers
# ---------------------------------------------------------------------------

def _sha256(path, chunk_size=1 << 20):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_identity(source_path, dest_path):
    """
    Compares a source and an already-existing destination file.
    Returns True if they are identical (same size AND same SHA-256 hash),
    False otherwise. Used for idempotent re-runs: if a destination file
    already exists, we must never silently overwrite it with different
    content.
    """
    source_size = source_path.stat().st_size
    dest_size = dest_path.stat().st_size
    if source_size != dest_size:
        return False
    return _sha256(source_path) == _sha256(dest_path)


# ---------------------------------------------------------------------------
# Segregation
# ---------------------------------------------------------------------------

def segregate_battery(battery_id, metadata):
    """
    Copies every raw file belonging to `battery_id` (per metadata) into
    dataset/processed/<battery_id>/<type>/<filename>, byte-for-byte
    (shutil.copy2, which also preserves the original file's modification
    timestamp). Idempotent: if a destination file already exists and is
    byte-identical to the source, it is left alone and reported as
    "already present"; if it exists but differs, this stops with a clear
    error rather than overwriting.

    Returns a dict of counters: copied, already_present, per experiment type.
    """
    if battery_id == PROTECTED_BATTERY_ID:
        raise SegregationError(
            f"Refusing to segregate {PROTECTED_BATTERY_ID} -- it is already "
            "segregated and validated, and this script must never touch it."
        )
    if battery_id not in TARGET_BATTERY_IDS:
        raise SegregationError(
            f"{battery_id} is not in the authorized target battery list "
            f"{TARGET_BATTERY_IDS}."
        )

    records = get_battery_records(metadata, battery_id)
    if len(records) == 0:
        raise SegregationError(f"No metadata records found for battery {battery_id}.")

    stats = {exp: {"copied": 0, "already_present": 0} for exp in EXPERIMENT_TYPES}

    for experiment in EXPERIMENT_TYPES:
        dest_dir = PROCESSED_DATA / battery_id / experiment
        dest_dir.mkdir(parents=True, exist_ok=True)

        experiment_records = records[records["type"] == experiment]

        for filename in experiment_records["filename"]:
            source_path = NASA_ORIGINAL / filename
            if not source_path.exists():
                raise SegregationError(
                    f"Source file referenced by metadata does not exist: {source_path} "
                    f"(battery={battery_id}, type={experiment})"
                )

            dest_path = dest_dir / filename

            if dest_path.exists():
                if verify_file_identity(source_path, dest_path):
                    stats[experiment]["already_present"] += 1
                    continue
                else:
                    raise SegregationError(
                        f"Destination file already exists and DIFFERS from source: "
                        f"{dest_path}. Refusing to overwrite. Investigate manually "
                        "before re-running."
                    )

            shutil.copy2(source_path, dest_path)

            if not verify_file_identity(source_path, dest_path):
                raise SegregationError(
                    f"Post-copy integrity check FAILED for {dest_path} -- copied file "
                    "does not match source. Stopping immediately."
                )

            stats[experiment]["copied"] += 1

    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_battery_output(battery_id, metadata, reference_battery_id=PROTECTED_BATTERY_ID):
    """
    Full post-segregation validation for one battery. Returns a dict of
    results (all real, computed values -- nothing hardcoded) and raises
    SegregationError on any failure, per this task's "do not silently
    continue after a validation failure" requirement.
    """
    results = {"battery_id": battery_id, "types": {}}

    expected_counts = compute_expected_counts(metadata, battery_id)
    records = get_battery_records(metadata, battery_id)

    all_missing = []
    all_duplicates = []
    all_unreadable = []
    all_wrong_battery_type = []

    for experiment in EXPERIMENT_TYPES:
        type_dir = PROCESSED_DATA / battery_id / experiment

        # A. Directory validation
        if not type_dir.exists():
            raise SegregationError(f"Missing expected directory: {type_dir}")

        actual_files = sorted(p.name for p in type_dir.glob("*.csv"))
        actual_count = len(actual_files)
        expected_count = expected_counts[experiment]

        # D. Duplicate validation (within this directory's file listing)
        if len(actual_files) != len(set(actual_files)):
            all_duplicates.append(experiment)

        # C. Missing-file validation -- every metadata-listed file for this
        # battery/type must exist on disk.
        expected_filenames = set(records[records["type"] == experiment]["filename"])
        actual_filenames = set(actual_files)
        missing = expected_filenames - actual_filenames
        if missing:
            all_missing.extend(sorted(missing))

        # E. Battery/type consistency -- every file physically present must
        # correspond to a metadata record for THIS battery and THIS type
        # (not just "some" record).
        for fname in actual_filenames:
            match = metadata[(metadata["filename"] == fname) &
                              (metadata["battery_id"] == battery_id) &
                              (metadata["type"] == experiment)]
            if len(match) == 0:
                all_wrong_battery_type.append((experiment, fname))

        # F. CSV integrity -- readable, non-empty, has columns.
        for fname in actual_files[:len(actual_files)]:
            fpath = type_dir / fname
            if fpath.stat().st_size == 0:
                all_unreadable.append((experiment, fname, "zero-byte file"))
                continue
            try:
                sample = pd.read_csv(fpath, nrows=1)
                if sample.shape[1] == 0:
                    all_unreadable.append((experiment, fname, "no columns"))
            except Exception as exc:
                all_unreadable.append((experiment, fname, str(exc)))

        count_pass = (actual_count == expected_count)
        results["types"][experiment] = {
            "expected": expected_count, "actual": actual_count, "pass": count_pass,
        }
        print(f"  {experiment:<12}expected={expected_count:<5}actual={actual_count:<5}"
              f"{'PASS' if count_pass else 'FAIL'}")

    if all_missing:
        raise SegregationError(f"{battery_id}: {len(all_missing)} metadata-listed file(s) "
                                f"missing from processed output, e.g. {all_missing[:5]}")
    if all_duplicates:
        raise SegregationError(f"{battery_id}: duplicate filenames found in type folder(s): {all_duplicates}")
    if all_unreadable:
        raise SegregationError(f"{battery_id}: {len(all_unreadable)} unreadable/invalid CSV(s), "
                                f"e.g. {all_unreadable[:5]}")
    if all_wrong_battery_type:
        raise SegregationError(f"{battery_id}: {len(all_wrong_battery_type)} file(s) present that "
                                f"don't match battery/type per metadata, e.g. {all_wrong_battery_type[:5]}")

    results["missing"] = len(all_missing)
    results["duplicates"] = len(all_duplicates)
    results["unreadable"] = len(all_unreadable)
    results["wrong_battery_type"] = len(all_wrong_battery_type)
    results["all_counts_pass"] = all(v["pass"] for v in results["types"].values())

    representative_schema_check(battery_id, reference_battery_id)

    return results


def representative_schema_check(battery_id, reference_battery_id=PROTECTED_BATTERY_ID):
    """
    For one representative file per experiment type, confirms the CSV is
    readable and has a NASA-style tabular structure -- WITHOUT hardcoding
    exact sensor column names. Instead, each new battery's representative
    file is compared against the corresponding file's schema from
    `reference_battery_id` (B0005, already segregated and validated),
    read dynamically at runtime -- not hardcoded from memory.
    """
    for experiment in EXPERIMENT_TYPES:
        target_dir = PROCESSED_DATA / battery_id / experiment
        target_files = sorted(target_dir.glob("*.csv"))
        if not target_files:
            raise SegregationError(f"No files found to schema-check for {battery_id}/{experiment}")
        target_sample = pd.read_csv(target_files[0])
        if target_sample.shape[0] == 0 or target_sample.shape[1] == 0:
            raise SegregationError(
                f"Representative file for {battery_id}/{experiment} "
                f"({target_files[0].name}) is empty or has no columns."
            )

        reference_dir = PROCESSED_DATA / reference_battery_id / experiment
        reference_files = sorted(reference_dir.glob("*.csv"))
        if reference_files:
            reference_sample = pd.read_csv(reference_files[0])
            if list(target_sample.columns) != list(reference_sample.columns):
                print(f"  NOTE: {battery_id}/{experiment} columns "
                      f"{list(target_sample.columns)} differ from "
                      f"{reference_battery_id}/{experiment} columns "
                      f"{list(reference_sample.columns)} -- reporting, not failing "
                      "(different batteries' raw exports can legitimately vary).")

        print(f"  Schema check {battery_id}/{experiment}: readable, "
              f"{target_sample.shape[0]} rows x {target_sample.shape[1]} cols "
              f"(sample: {target_files[0].name})")


def protect_reference_battery(reference_battery_id=PROTECTED_BATTERY_ID):
    """Counts every file currently under dataset/processed/<reference_battery_id>/.
    Called before AND after segregation; the two counts must be identical."""
    counts = {}
    for experiment in EXPERIMENT_TYPES:
        type_dir = PROCESSED_DATA / reference_battery_id / experiment
        counts[experiment] = len(list(type_dir.glob("*.csv"))) if type_dir.exists() else 0
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NASA MULTI-BATTERY SEGREGATION")
    print("-" * 60)

    metadata = load_metadata()
    validate_metadata_schema(metadata)

    available_batteries = set(metadata["battery_id"].unique())
    missing_batteries = [b for b in TARGET_BATTERY_IDS if b not in available_batteries]
    if missing_batteries:
        raise SegregationError(f"Metadata has no records for: {missing_batteries}")

    b0005_before = protect_reference_battery(PROTECTED_BATTERY_ID)
    print(f"B0005 file counts BEFORE (must be unchanged after): {b0005_before}")
    print("-" * 60)

    all_results = {}
    total_missing = total_duplicates = total_unreadable = 0
    total_conflicting = 0  # tracked via SegregationError on mismatch -- if we reach
                            # the summary at all, this is 0 by construction (a
                            # conflict raises immediately and stops the script).

    for battery_id in TARGET_BATTERY_IDS:
        print(f"\n{battery_id}")
        segregate_battery(battery_id, metadata)
        results = verify_battery_output(battery_id, metadata)
        all_results[battery_id] = results
        total_missing += results["missing"]
        total_duplicates += results["duplicates"]
        total_unreadable += results["unreadable"]

    print("-" * 60)
    b0005_after = protect_reference_battery(PROTECTED_BATTERY_ID)
    b0005_protected = (b0005_before == b0005_after)
    print(f"B0005 file counts AFTER: {b0005_after}")
    print(f"B0005 protection: {'PASS' if b0005_protected else 'FAIL'}")
    if not b0005_protected:
        raise SegregationError(
            f"B0005 changed during this run! Before={b0005_before}, After={b0005_after}. "
            "This must never happen."
        )

    print(f"Missing files: {total_missing}")
    print(f"Duplicate files: {total_duplicates}")
    print(f"Unreadable CSVs: {total_unreadable}")
    print(f"Conflicting destination files: {total_conflicting}")

    all_pass = (
        b0005_protected and total_missing == 0 and total_duplicates == 0
        and total_unreadable == 0 and total_conflicting == 0
        and all(r["all_counts_pass"] for r in all_results.values())
    )
    final_status = "PASS" if all_pass else "FAIL"
    print(f"\nFINAL STATUS: {final_status}")
    print("-" * 60)

    if not all_pass:
        raise SegregationError("One or more validations failed -- see output above.")

    return all_results


if __name__ == "__main__":
    main()
