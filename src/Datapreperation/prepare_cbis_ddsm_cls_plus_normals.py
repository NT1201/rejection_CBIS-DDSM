
#!/usr/bin/env python3
"""
CBIS-DDSM classification prep (FULL IMAGES + NORMAL class).

Builds a 3-class dataset:
  NORMAL (0), BENIGN (1), MALIGNANT (2)

Sources:
- Case CSVs (mass_* and calc_*) provide BENIGN/MALIGNANT with 'image file path' (DICOM) and 'patient_id'.
- dicom_info.csv maps DICOM <series_uid>/<sop>.dcm to JPEGs under 'jpeg/<series_uid>/*.jpg'.
- meta.csv (optional) may map series_uid → patient_id; if missing, we use series_uid as a pseudo patient_id for NORMALs.

Key idea:
- "Normals" are series present in dicom_info.csv but **absent** from the case CSVs.
- We pick one JPEG per series (first lexicographically) as the representative image.

Output:
out/
  train/images + labels.csv
  val/images   + labels.csv   (if --val_ratio>0)
  test/images  + labels.csv
  manifest.csv
  class_counts.json
  class_mapping.json          # {"NORMAL":0,"BENIGN":1,"MALIGNANT":2}

Usage example:
  python prepare_cbis_ddsm_cls_plus_normals.py --root "C:/ThisMyFinal/Rawdata" --out "./cbis_cls3" --val_ratio 0.15 --copy --normal_ratio 1.0

Arguments:
  --normal_ratio   float  Downsample NORMAL count per split relative to lesion count (default 1.0 = same count as total benign+malignant in that split).
  --seed           int    Random seed (patient-level sampling).
  --csv_dir        str    Override CSV directory (default: ROOT/csv)
  --copy           flag   Copy instead of symlink (Windows-friendly)
"""

import argparse, os, sys, json, shutil, random, re
from collections import defaultdict, Counter
import pandas as pd

CLASS_MAP = {"NORMAL":0, "BENIGN":1, "MALIGNANT":2}
PATHOLOGY_MAP_2C = {'BENIGN': CLASS_MAP["BENIGN"],
                    'BENIGN_WITHOUT_CALLBACK': CLASS_MAP["BENIGN"],
                    'MALIGNANT': CLASS_MAP["MALIGNANT"]}
CSV_NAMES = {
    'train': ['mass_case_description_train_set.csv','calc_case_description_train_set.csv'],
    'test':  ['mass_case_description_test_set.csv','calc_case_description_test_set.csv'],
}

def norm(p): return str(p).strip().replace('\\','/')

def ensure_rel_from_jpeg(p):
    s = norm(p); low = s.lower()
    idx = low.find('jpeg/')
    return s[idx:] if idx != -1 else s

def series_uid_from_dicom_file_path(p):
    s = norm(p)
    parts = s.split('/')
    if len(parts)>=2 and parts[-1].endswith('.dcm'):
        return parts[-2]
    # fallback: last UID-like token
    for token in reversed(parts):
        if re.fullmatch(r'(?:\d+\.)+\d+', token):
            return token
    return None

def load_series_to_jpegs(dicom_info_csv):
    df = pd.read_csv(dicom_info_csv)
    if 'file_path' not in df.columns or 'image_path' not in df.columns:
        raise ValueError("dicom_info.csv must contain 'file_path' and 'image_path'.")
    df['series_uid'] = df['file_path'].astype(str).apply(series_uid_from_dicom_file_path)
    df['image_path'] = df['image_path'].astype(str).apply(ensure_rel_from_jpeg)
    series2jpegs = df.groupby('series_uid')['image_path'].apply(lambda s: sorted(set(s))).to_dict()
    return series2jpegs

def best_jpeg_for_series(jpegs):
    jpegs = sorted(jpegs)
    return jpegs[0]

def link_or_copy(src, dst, do_copy=False):
    if os.path.exists(dst): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            rel = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(rel, dst)
        except Exception:
            shutil.copy2(src, dst)

def read_lesion_rows(csv_dir):
    """Return dicts: rows_train, rows_test, lesion_series_uids set"""
    def read_case_rows(csv_path):
        df = pd.read_csv(csv_path)
        req = ['image file path','pathology','patient_id']
        for c in req:
            if c not in df.columns:
                raise ValueError(f"{csv_path} missing required column: {c}")
        df = df.dropna(subset=req).copy()
        rows = []
        series_uids = set()
        for _, r in df.iterrows():
            su = series_uid_from_dicom_file_path(r['image file path'])
            patho = str(r['pathology']).strip().upper()
            pid = str(r['patient_id']).strip()
            if patho not in PATHOLOGY_MAP_2C: 
                continue
            rows.append((su, PATHOLOGY_MAP_2C[patho], pid))
            series_uids.add(su)
        return rows, series_uids

    rows_train, rows_test = [], []
    lesion_series = set()
    for name in CSV_NAMES['train']:
        p = os.path.join(csv_dir, name)
        if not os.path.isfile(p):
            print(f"[WARN] Missing CSV: {p}", file=sys.stderr); continue
        r, s = read_case_rows(p)
        rows_train.extend(r); lesion_series |= s
    for name in CSV_NAMES['test']:
        p = os.path.join(csv_dir, name)
        if not os.path.isfile(p):
            print(f"[WARN] Missing CSV: {p}", file=sys.stderr); continue
        r, s = read_case_rows(p)
        rows_test.extend(r); lesion_series |= s
    return rows_train, rows_test, lesion_series

def maybe_map_series_to_patient_via_meta(csv_dir):
    """Try to map series_uid -> patient_id using meta.csv if available; else return empty dict."""
    meta_path = os.path.join(csv_dir, "meta.csv")
    mapping = {}
    if not os.path.isfile(meta_path):
        return mapping
    try:
        df = pd.read_csv(meta_path)
        # Try common column names
        candidates = [("SeriesInstanceUID","PatientID"),
                      ("series_instance_uid","patient_id"),
                      ("SeriesInstanceUID","patient_id"),
                      ("series_uid","patient_id")]
        for c_series, c_pid in candidates:
            if c_series in df.columns and c_pid in df.columns:
                tmp = df[[c_series, c_pid]].dropna()
                for _, r in tmp.iterrows():
                    su = str(r[c_series]).strip()
                    pid = str(r[c_pid]).strip()
                    mapping[su] = pid
                if mapping:
                    print(f"[INFO] meta.csv mapping: {len(mapping)} series_uid→patient_id")
                    return mapping
    except Exception as e:
        print(f"[WARN] Failed to parse meta.csv: {e}", file=sys.stderr)
    return mapping

def split_patients(patient_ids, ratios, seed=42):
    """Split a set of patient_ids into (train,val,test) by ratios (sum<=1)."""
    rnd = random.Random(seed)
    ids = list(patient_ids); rnd.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val   = int(round(n * ratios[1]))
    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train+n_val])
    test_ids  = set(ids[n_train+n_val:])
    return train_ids, val_ids, test_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder with 'csv/' and 'jpeg/'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv_dir", default=None)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal_ratio", type=float, default=1.0, help="Cap NORMAL count per split to normal_ratio * lesion_count(split)")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    csv_dir = args.csv_dir or os.path.join(args.root, "csv")
    dicom_info_csv = os.path.join(csv_dir, "dicom_info.csv")
    if not os.path.isdir(csv_dir):
        print(f"[ERROR] CSV dir not found: {csv_dir}", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(os.path.join(args.root, "jpeg")):
        print(f"[ERROR] JPEG dir not found under: {args.root}", file=sys.stderr); sys.exit(1)

    # Mappings and lesion rows
    series2jpegs = load_series_to_jpegs(dicom_info_csv)
    rows_train_les, rows_test_les, lesion_series = read_lesion_rows(csv_dir)
    print(f"[INFO] Lesion rows: train={len(rows_train_les)} test={len(rows_test_les)} | unique series with lesions={len(lesion_series)}")
    # Map lesion rows to (jpg_rel, label, patient_id)
    def rows_to_jpg(rows):
        out = []
        miss = 0
        for su, lbl, pid in rows:
            jpegs = series2jpegs.get(su)
            if not jpegs:
                miss += 1; continue
            out.append((best_jpeg_for_series(jpegs), lbl, pid))
        return out, miss
    rows_train_labeled, miss_tr = rows_to_jpg(rows_train_les)
    rows_test_labeled,  miss_te = rows_to_jpg(rows_test_les)
    if miss_tr or miss_te:
        print(f"[INFO] Missing series→jpeg: train={miss_tr}, test={miss_te}")

    # Build NORMAL candidates: all series in dicom_info that are not lesion_series
    meta_map = maybe_map_series_to_patient_via_meta(csv_dir)
    normal_series = [su for su in series2jpegs.keys() if su not in lesion_series and su is not None]
    # Map normals to patient_id (from meta if possible; else series_uid as pseudo pid)
    normal_rows = [(best_jpeg_for_series(series2jpegs[su]), CLASS_MAP["NORMAL"], meta_map.get(su, f"S_{su}")) for su in normal_series if su in series2jpegs]
    print(f"[INFO] Normal series found: {len(normal_rows)} (before sampling)")

    # Determine test ratio based on lesion patients
    les_train_pids = set(pid for _,_,pid in rows_train_labeled)
    les_test_pids  = set(pid for _,_,pid in rows_test_labeled)
    total_les_pids = len(les_train_pids | les_test_pids)
    test_ratio = (len(les_test_pids) / total_les_pids) if total_les_pids else 0.2
    train_ratio = 1.0 - test_ratio
    # We'll split normals into train+val vs test using same patient-level proportion
    normal_pids = sorted(set(pid for _,_,pid in normal_rows))
    rnd = random.Random(args.seed)
    rnd.shuffle(normal_pids)
    n_test_norm = int(round(len(normal_pids) * test_ratio))
    test_norm_ids = set(normal_pids[:n_test_norm])
    trainval_norm_ids = set(normal_pids[n_test_norm:])

    # Now within trainval normals, create val split by args.val_ratio (patient-level)
    n_val_norm = int(round(len(trainval_norm_ids) * args.val_ratio))
    val_norm_ids = set(list(trainval_norm_ids)[:n_val_norm])
    train_norm_ids = trainval_norm_ids - val_norm_ids

    # Partition normals by split
    normals_train = [r for r in normal_rows if r[2] in train_norm_ids]
    normals_val   = [r for r in normal_rows if r[2] in val_norm_ids]
    normals_test  = [r for r in normal_rows if r[2] in test_norm_ids]

    # Cap normals per split according to --normal_ratio relative to lesion counts
    def cap_normals(norm_rows_split, lesion_rows_split):
        if args.normal_ratio <= 0:
            return []
        cap = int(round(len(lesion_rows_split) * args.normal_ratio))
        if len(norm_rows_split) <= cap:
            return norm_rows_split
        # sample by patient groups for stability
        per_pid = defaultdict(list)
        for r in norm_rows_split:
            per_pid[r[2]].append(r)
        pids = list(per_pid.keys()); rnd.shuffle(pids)
        out = []
        for pid in pids:
            if len(out) >= cap: break
            out.extend(per_pid[pid])
        return out[:cap]

    normals_train = cap_normals(normals_train, rows_train_labeled)
    normals_val   = cap_normals(normals_val, rows_train_labeled if args.val_ratio>0 else rows_test_labeled)
    normals_test  = cap_normals(normals_test, rows_test_labeled)

    # Merge splits
    rows_train = rows_train_labeled + normals_train
    rows_val   = (rows_test_labeled*0)  # placeholder, not used
    if args.val_ratio > 0:
        # Make a patient-level val split for lesions
        per_pid_les = defaultdict(list)
        for r in rows_train_labeled:
            per_pid_les[r[2]].append(r)
        les_pids = list(per_pid_les.keys()); rnd.shuffle(les_pids)
        n_val_les = int(round(len(les_pids) * args.val_ratio))
        val_les_ids = set(les_pids[:n_val_les])
        train_les_ids = set(les_pids[n_val_les:])
        train_les = [r for r in rows_train_labeled if r[2] in train_les_ids]
        val_les   = [r for r in rows_train_labeled if r[2] in val_les_ids]
        # Combine with normals
        rows_train = train_les + normals_train
        rows_val   = val_les   + normals_val
    rows_test  = rows_test_labeled + normals_test

    # Shuffle each split to mix normals/lesions
    rnd.shuffle(rows_train)
    if args.val_ratio > 0: rnd.shuffle(rows_val)
    rnd.shuffle(rows_test)

    # Write outputs
    os.makedirs(args.out, exist_ok=True)
    def write_split(rows, split):
        images_dir = os.path.join(args.out, split, "images")
        os.makedirs(images_dir, exist_ok=True)
        labels, counts = [], Counter()
        for i, (jpg_rel, lbl, pid) in enumerate(rows):
            rel = ensure_rel_from_jpeg(jpg_rel)
            src = os.path.join(args.root, rel)
            if not os.path.isfile(src): 
                continue
            dst = os.path.join(images_dir, f"{split}_{i:06d}.jpg")
            link_or_copy(src, dst, do_copy=args.copy)
            labels.append({"image": f"images/{os.path.basename(dst)}", "label": int(lbl), "patient_id": pid})
            counts[int(lbl)] += 1
        pd.DataFrame(labels).to_csv(os.path.join(args.out, split, "labels.csv"), index=False)
        return counts, labels

    counts = {}
    man_rows = []
    c_train, train_lbls = write_split(rows_train, "train")
    counts["train"] = {"NORMAL": int(c_train.get(CLASS_MAP["NORMAL"],0)),
                       "BENIGN": int(c_train.get(CLASS_MAP["BENIGN"],0)),
                       "MALIGNANT": int(c_train.get(CLASS_MAP["MALIGNANT"],0)),
                       "total": sum(c_train.values())}
    if args.val_ratio > 0:
        c_val, val_lbls = write_split(rows_val, "val")
        counts["val"] = {"NORMAL": int(c_val.get(CLASS_MAP["NORMAL"],0)),
                         "BENIGN": int(c_val.get(CLASS_MAP["BENIGN"],0)),
                         "MALIGNANT": int(c_val.get(CLASS_MAP["MALIGNANT"],0)),
                         "total": sum(c_val.values())}
    else:
        val_lbls = []
    c_test, test_lbls = write_split(rows_test, "test")
    counts["test"] = {"NORMAL": int(c_test.get(CLASS_MAP["NORMAL"],0)),
                      "BENIGN": int(c_test.get(CLASS_MAP["BENIGN"],0)),
                      "MALIGNANT": int(c_test.get(CLASS_MAP["MALIGNANT"],0)),
                      "total": sum(c_test.values())}

    # Manifest
    def add_manifest(lbls, split):
        for r in lbls:
            man_rows.append({"split": split, "image": f"{split}/{r['image']}", "label": r["label"], "patient_id": r["patient_id"]})
    add_manifest(train_lbls, "train")
    add_manifest(val_lbls, "val") if val_lbls else None
    add_manifest(test_lbls, "test")

    pd.DataFrame(man_rows).to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    with open(os.path.join(args.out, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    with open(os.path.join(args.out, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)

    print("[DONE] Prepared 3-class dataset at:", args.out)
    print(json.dumps(counts, indent=2))
    print("Class mapping saved to class_mapping.json:", CLASS_MAP)

if __name__ == "__main__":
    main()
