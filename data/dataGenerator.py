import pandas as pd
import numpy as np
import random
import hashlib
from datetime import datetime, timedelta

# -------------------------
# CONFIG (tweak these)
# -------------------------
SEED = 42
NUM_INSPECTIONS = 20000
NUM_WORKERS = 200
NUM_INSPECTORS = 50
NUM_LANES = 300
NUM_VENDORS = 30

# Make anomalies rare by default (e.g., 0.01 = 1%)
ANOMALY_RATE = 0.03

# Control relative frequency of specific anomaly types (only used when selecting which anomalies to inject)
ANOMALY_TYPE_WEIGHTS = {
    "overdue": 1,
    "gps_mismatch": 1,
    "parts_mismatch": 1,
    "photo_anomaly": 1,
    "backdated": 0.8,
    "impossible_travel": 0.8,
    "rubber_stamp": 0.8,
    "privilege_abuse": 0.5,
    "collusion": 0.6
}

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 9, 1)

# -------------------------
# Setup RNG
# -------------------------
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Helpers
# -------------------------
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def random_photo_hash():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:8]

def random_ip():
    return f"{random.randint(10, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def clip_positive(x, min_val=0.0):
    return float(max(x, min_val))

def choose_anomaly_types():
    # choose 1 to 3 anomaly types based on weights (allow multiple)
    types = list(ANOMALY_TYPE_WEIGHTS.keys())
    weights = np.array([ANOMALY_TYPE_WEIGHTS[t] for t in types], dtype=float)
    weights /= weights.sum()
    k = random.choice([1,1,1,2])  # most anomalies get just 1, sometimes 2
    chosen = list(np.random.choice(types, size=k, replace=False, p=weights))
    return chosen

# -------------------------
# Base normal generators
# -------------------------
def generate_normal_row(i):
    inspection_id = i + 1
    worker_id = random.randint(1, NUM_WORKERS)
    inspector_id = random.randint(1, NUM_INSPECTORS)
    lane_id = random.randint(1, NUM_LANES)
    vendor_id = random.randint(1, NUM_VENDORS)

    # Dates: for normals, submission is usually near due_date (-2 to +1 days)
    due_date = random_date(start_date, end_date)
    submission_time = due_date + timedelta(days=random.randint(-2, 1), 
                                           seconds=random.randint(0, 86400))

    # Status: mostly completed
    status = random.choices(["completed", "pending", "rejected"], weights=[0.85, 0.1, 0.05])[0]

    # GPS distances (typical small distances)
    gps_worker = abs(np.random.normal(20, 10))
    gps_inspector = abs(np.random.normal(12, 8))
    gps_distance_to_asset = abs(np.random.normal(8, 6))

    # Parts & stock (normally consistent)
    parts_reported = random.randint(0, 5)
    # usually inventory_withdrawal equals parts_reported or off-by-1 occasionally
    inventory_withdrawal = parts_reported - random.choice([0, 0, 0, 1])
    parts_inventory_diff = parts_reported - inventory_withdrawal

    # Photos (normal cases submit photos, similarity low)
    photo_hash = random_photo_hash()
    photo_similarity_score = np.random.beta(2, 20)  # most low similarity
    num_photos_submitted = random.choices([1,2,3], weights=[0.6,0.3,0.1])[0]

    # Editing & backdating (rare edits)
    edit_count = random.choices([0,1,2], weights=[0.85,0.13,0.02])[0]
    time_since_last_edit = random.randint(0, 3000) if edit_count > 0 else 0
    backdated_flag = 0

    # Timing
    time_to_submit = random.randint(30, 3600)
    time_between_inspections = random.randint(30, 7200)
    hour_of_day = submission_time.hour
    day_of_week = submission_time.weekday()

    # Device/Network
    device_id = f"device_{random.randint(1, 200)}"
    ip_address = random_ip()
    app_version = f"v{random.randint(1,3)}.{random.randint(0,9)}"
    session_duration = random.randint(20, 7200)
    session_count = random.randint(1, 10)

    # Inspector behaviour
    implied_velocity = round(np.random.normal(12, 8), 2)
    validation_bypasses = random.choices([0,1,2], weights=[0.9,0.08,0.02])[0]
    consensus_score = round(np.random.beta(5, 2), 3)  # normal high consensus

    # Inspector-vendor network features
    inspector_vendor_edge_weight = random.randint(1, 20)
    inspector_centrality = round(np.random.beta(2, 5), 3)

    return {
        "inspection_id": inspection_id, "worker_id": worker_id, "inspector_id": inspector_id,
        "lane_id": lane_id, "vendor_id": vendor_id, "maintenance_due_date": due_date,
        "submission_time": submission_time, "status": status,
        "gps_distance_worker_to_lane": clip_positive(gps_worker),
        "gps_distance_inspector_to_lane": clip_positive(gps_inspector),
        "gps_distance_to_asset": clip_positive(gps_distance_to_asset),
        "parts_reported": parts_reported, "inventory_withdrawal": inventory_withdrawal,
        "parts_inventory_diff": parts_inventory_diff, "photo_hash": photo_hash,
        "photo_similarity_score": float(photo_similarity_score),
        "num_photos_submitted": num_photos_submitted, "edit_count": edit_count,
        "time_since_last_edit": time_since_last_edit, "backdated_flag": backdated_flag,
        "time_to_submit": time_to_submit, "time_between_inspections": time_between_inspections,
        "hour_of_day": hour_of_day, "day_of_week": day_of_week,
        "device_id": device_id, "ip_address": ip_address, "app_version": app_version,
        "session_duration": session_duration, "session_count": session_count,
        "implied_velocity": clip_positive(implied_velocity),
        "validation_bypasses": validation_bypasses, "consensus_score": consensus_score,
        "inspector_vendor_edge_weight": inspector_vendor_edge_weight,
        "inspector_centrality": inspector_centrality,
        # placeholders; will be updated if anomalies are injected
        "anomaly_flag": 0, "anomaly_type": "normal"
    }

# -------------------------
# Anomaly perturbations (mutate the normal row)
# -------------------------
def inject_overdue(r):
    # Make submission late and status not completed
    r["submission_time"] = r["maintenance_due_date"] + timedelta(days=random.randint(2, 20))
    r["status"] = random.choice(["pending", "rejected"])
    return r

def inject_gps_mismatch(r):
    # blow up GPS distances
    r["gps_distance_worker_to_lane"] = float(abs(np.random.normal(150, 40)))
    r["gps_distance_inspector_to_lane"] = float(abs(np.random.normal(130, 45)))
    return r

def inject_parts_mismatch(r):
    # create clear mismatch where reported != withdrawn (negative or big diff)
    r["parts_reported"] = random.randint(0, 8)
    r["inventory_withdrawal"] = r["parts_reported"] - random.choice([1,2,3, -1])
    r["parts_inventory_diff"] = r["parts_reported"] - r["inventory_withdrawal"]
    return r

def inject_photo_anomaly(r):
    # no photos or identical photos (very high similarity)
    if random.random() < 0.5:
        r["num_photos_submitted"] = 0
        r["photo_similarity_score"] = 1.0
    else:
        r["num_photos_submitted"] = random.choice([1])
        r["photo_similarity_score"] = float(np.random.uniform(0.98, 1.0))
    return r

def inject_backdated(r):
    # submission earlier than due_date by many days OR mark backdated_flag
    r["submission_time"] = r["maintenance_due_date"] - timedelta(days=random.randint(5, 60))
    r["backdated_flag"] = 1
    r["edit_count"] = max(r["edit_count"], random.randint(3, 8))
    return r

def inject_impossible_travel(r):
    r["implied_velocity"] = float(np.random.uniform(130, 400))
    return r

def inject_rubber_stamp(r):
    r["time_to_submit"] = random.randint(1, 10)
    r["status"] = "completed"
    return r

def inject_privilege_abuse(r):
    r["validation_bypasses"] = random.randint(3, 8)
    return r

def inject_collusion(r):
    r["inspector_vendor_edge_weight"] = random.randint(41, 100)
    r["consensus_score"] = float(np.random.uniform(0.0, 0.15))
    r["inspector_centrality"] = float(np.random.uniform(0.8, 1.0))
    return r

ANOMALY_INJECTORS = {
    "overdue": inject_overdue,
    "gps_mismatch": inject_gps_mismatch,
    "parts_mismatch": inject_parts_mismatch,
    "photo_anomaly": inject_photo_anomaly,
    "backdated": inject_backdated,
    "impossible_travel": inject_impossible_travel,
    "rubber_stamp": inject_rubber_stamp,
    "privilege_abuse": inject_privilege_abuse,
    "collusion": inject_collusion
}

# -------------------------
# Generation loop
# -------------------------
rows = []
for i in range(NUM_INSPECTIONS):
    r = generate_normal_row(i)

    if random.random() < ANOMALY_RATE:
        # This row is anomalous: select 1-3 anomaly types and apply injectors
        types = choose_anomaly_types()
        for t in types:
            injector = ANOMALY_INJECTORS.get(t)
            if injector:
                r = injector(r)
        r_types = ";".join(types)
        r["anomaly_flag"] = 1
        r["anomaly_type"] = r_types
    else:
        r["anomaly_flag"] = 0
        r["anomaly_type"] = "normal"

    # Recompute dependent derived values (defensive)
    r["parts_inventory_diff"] = r["parts_reported"] - r["inventory_withdrawal"]
    r["hour_of_day"] = r["submission_time"].hour
    r["day_of_week"] = r["submission_time"].weekday()
    # Ensure numeric fields are plain Python floats/ints for CSV cleanliness
    rows.append(r)

# -------------------------
# DataFrame & Save
# -------------------------
df = pd.DataFrame(rows)

# Optional: shuffle dataset rows (so anomalies not all clustered)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

filename = "synthetic_inspections_labeled.csv"
df.to_csv(filename, index=False)
print(f"✅ Dataset generated → {filename}")

# -------------------------
# Quick checks (printed)
# -------------------------
print("\n--- Quick distribution checks ---")
counts = df['anomaly_flag'].value_counts()
print("Anomaly counts:\n", counts.to_string())
pct = df['anomaly_flag'].value_counts(normalize=True).mul(100).round(3)
print("\nAnomaly %:\n", pct.to_string())
print("\nTop anomaly types (head):")
print(df.loc[df['anomaly_flag']==1, 'anomaly_type'].value_counts().head(10))
print("\nSample anomalous rows:")
print(df.loc[df['anomaly_flag']==1].head(5).T)
