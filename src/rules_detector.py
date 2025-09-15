import os
import json
from collections import Counter
import pandas as pd
import numpy as np

# --- CONFIG / THRESHOLDS (tuneable) ---
INPUT_CSV = "data/synthetic_inspections_labeled.csv"
OUT_DIR = "outputs"
GPS_DISTANCE_THRESHOLD = 100.0           # meters
CONSECUTIVE_PAIR_THRESHOLD = 5           # same inspector-worker in a row
PHOTO_SIMILARITY_THRESHOLD = 0.95
TIME_TO_SUBMIT_RUBBERSTAMP_SEC = 15      # too-fast submissions
IMPOSSIBLE_VELOCITY_KMPH = 120.0
VALIDATION_BYPASS_THRESHOLD = 2
COLLUSION_EDGE_WEIGHT = 40
COLLUSION_CONSENSUS_MAX = 0.2


# --- UTILS ---
def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_df(path):
    df = pd.read_csv(path, parse_dates=["maintenance_due_date", "submission_time"])
    # safety: fill missing numeric cols
    numeric_cols = [
        "gps_distance_worker_to_lane","gps_distance_inspector_to_lane","gps_distance_to_asset",
        "parts_reported","inventory_withdrawal","parts_inventory_diff",
        "photo_similarity_score","num_photos_submitted","edit_count","time_since_last_edit",
        "backdated_flag","time_to_submit","time_between_inspections","hour_of_day","day_of_week",
        "session_duration","session_count","implied_velocity","validation_bypasses","consensus_score",
        "inspector_vendor_edge_weight","inspector_centrality"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


# --- RULE CHECKS (returns list of string flags) ---
def check_overdue(row):
    try:
        if pd.notna(row["maintenance_due_date"]) and pd.notna(row["submission_time"]):
            if (row["maintenance_due_date"].to_pydatetime() < row["submission_time"].to_pydatetime()
                    and str(row.get("status","")).lower() != "completed"):
                return "overdue"
    except Exception:
        pass
    return None

def check_gps_mismatch(row):
    if row.get("gps_distance_worker_to_lane", 0) > GPS_DISTANCE_THRESHOLD or \
       row.get("gps_distance_inspector_to_lane", 0) > GPS_DISTANCE_THRESHOLD:
        return "gps_mismatch"
    return None

def check_parts_mismatch(row):
    if int(row.get("parts_inventory_diff", 0)) != 0:
        return "parts_mismatch"
    return None

def check_photo_anomaly(row):
    if int(row.get("num_photos_submitted", 0)) == 0 or float(row.get("photo_similarity_score", 0)) >= PHOTO_SIMILARITY_THRESHOLD:
        return "photo_anomaly"
    return None

def check_backdated(row):
    if int(row.get("backdated_flag", 0)) == 1 or int(row.get("edit_count", 0)) > 2:
        return "backdated"
    return None

def check_impossible_travel(row):
    if float(row.get("implied_velocity", 0)) > IMPOSSIBLE_VELOCITY_KMPH:
        return "impossible_travel"
    return None

def check_rubber_stamp(row):
    if int(row.get("time_to_submit", 99999)) < TIME_TO_SUBMIT_RUBBERSTAMP_SEC and str(row.get("status","")).lower() == "completed":
        return "rubber_stamp"
    return None

def check_privilege_bypass(row):
    if int(row.get("validation_bypasses", 0)) > VALIDATION_BYPASS_THRESHOLD:
        return "privilege_abuse"
    return None

def check_collusion(row):
    if int(row.get("inspector_vendor_edge_weight", 0)) > COLLUSION_EDGE_WEIGHT and float(row.get("consensus_score", 1)) < COLLUSION_CONSENSUS_MAX:
        return "collusion_suspected"
    return None


# --- HIGH LEVEL DETECTION FLOW ---
def detect_row_flags(df):
    flags_per_row = []
    for _, row in df.iterrows():
        flags = []
        for fn in (check_overdue, check_gps_mismatch, check_parts_mismatch,
                   check_photo_anomaly, check_backdated, check_impossible_travel,
                   check_rubber_stamp, check_privilege_bypass, check_collusion):
            f = fn(row)
            if f:
                flags.append(f)
        flags_per_row.append(flags)
    return flags_per_row


def detect_consecutive_pairs(df, threshold=CONSECUTIVE_PAIR_THRESHOLD):
    """Find inspections that are part of a run where the same (inspector,worker) appears `threshold` times in a row.
       We mark all inspections in the run as 'repeated_pair'."""
    df_sorted = df.sort_values(["inspector_id", "submission_time"]).reset_index(drop=True)
    repeated_flags = [""] * len(df_sorted)
    for insp_id, group in df_sorted.groupby("inspector_id"):
        prev_pair = None
        run_indices = []
        for idx, row in group.iterrows():
            pair = (row["inspector_id"], row["worker_id"])
            if pair == prev_pair:
                run_indices.append(idx)
            else:
                if len(run_indices) >= threshold:
                    for ridx in run_indices:
                        repeated_flags[ridx] = "repeated_pair"
                run_indices = [idx]
                prev_pair = pair
        if len(run_indices) >= threshold:
            for ridx in run_indices:
                repeated_flags[ridx] = "repeated_pair"
    key = df_sorted[["inspection_id"]].copy()
    key["repeated_pair_flag"] = repeated_flags
    map_df = key.set_index("inspection_id")["repeated_pair_flag"].to_dict()
    return df["inspection_id"].map(map_df).fillna("")


def detect_photo_reuse(df):
    reuse_flag = []
    counts = df.groupby("photo_hash")["lane_id"].nunique()
    reused_hashes = set(counts[counts > 1].index.tolist())
    for _, row in df.iterrows():
        if row.get("photo_hash") in reused_hashes:
            reuse_flag.append("photo_reuse_across_lanes")
        else:
            reuse_flag.append("")
    return reuse_flag


def inspector_summary(df, flags_col):
    summary = {}
    for insp, group in df.groupby("inspector_id"):
        all_flags = group[flags_col].apply(lambda x: x.split(";") if isinstance(x, str) and x else [])
        flat = [f for lst in all_flags for f in lst]
        c = Counter(flat)
        summary[str(insp)] = {
            "total_inspections": int(len(group)),
            "anomaly_count": int(sum(1 for x in group[flags_col] if isinstance(x, str) and x)),
            "top_flags": dict(c.most_common(10)),
            "median_time_to_submit": float(group["time_to_submit"].median() if "time_to_submit" in group.columns else 0),
            "avg_implied_velocity": float(group["implied_velocity"].mean() if "implied_velocity" in group.columns else 0)
        }
    return summary


# --- MAIN RUN ---
def run_detector(input_data=INPUT_CSV):
    """
    Accepts either:
    - str (CSV path)
    - pandas.DataFrame
    """
    ensure_out()

    if isinstance(input_data, str):
        df = load_df(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("run_detector() input must be CSV path or DataFrame")

    # rule checks
    df["flags_list"] = detect_row_flags(df)
    df["repeated_pair_flag"] = detect_consecutive_pairs(df)
    df["photo_reuse_flag"] = detect_photo_reuse(df)

    def combine_flags(r):
        combined = []
        combined += r["flags_list"] if isinstance(r["flags_list"], list) else []
        if r.get("repeated_pair_flag"):
            combined.append(r["repeated_pair_flag"])
        if r.get("photo_reuse_flag"):
            combined.append(r["photo_reuse_flag"])
        combined = list(dict.fromkeys(combined))
        return ";".join(combined) if combined else ""

    df["detected_flags"] = df.apply(combine_flags, axis=1)
    df["detected_anomaly_flag"] = df["detected_flags"].apply(lambda x: 1 if x else 0)

    # flagged rows
    flagged = df[df["detected_anomaly_flag"] == 1].copy()
    flagged.to_csv(os.path.join(OUT_DIR, "flags.csv"), index=False)

    # inspector summary
    summary = inspector_summary(df, "detected_flags")
    with open(os.path.join(OUT_DIR, "inspector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    insp_df = pd.DataFrame.from_dict(summary, orient="index").reset_index().rename(columns={"index": "inspector_id"})
    insp_df.to_csv(os.path.join(OUT_DIR, "inspector_summary.csv"), index=False)

    return df, flagged, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run rules-only anomaly detector")
    parser.add_argument("--input", help="input CSV path", default=INPUT_CSV)
    args = parser.parse_args()
    run_detector(args.input)
