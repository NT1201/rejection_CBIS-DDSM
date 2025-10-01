# C:\ThisMyFinal\src\Datapreperation\filter_cbis_to_bm.py
import os, csv, json, shutil, argparse
from pathlib import Path

def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        return list(r)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def write_csv(rows, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["image","label","patient_id"])
        w.writeheader()
        for row in rows: w.writerow(row)

def copy_or_link(src, dst, do_copy=False):
    ensure_dir(Path(dst).parent)
    if do_copy:
        if not Path(dst).exists():
            shutil.copy2(src, dst)
    else:
        # use hardlink if possible; fallback to copy
        try:
            if Path(dst).exists(): return
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def filter_split(src_split, dst_split, copy_imgs):
    # read labels
    rows = load_rows(Path(src_split, "labels.csv"))
    # map 3->2 classes and drop NORMAL
    label_map_3to2 = {"NORMAL": None, "BENIGN": 0, "MALIGNANT": 1}
    # accept both numeric and string labels
    inv = {"0":"NORMAL","1":"BENIGN","2":"MALIGNANT"}
    out_rows = []
    kept = {"BENIGN":0, "MALIGNANT":0}
    for r in rows:
        # decode label
        lab = r["label"].strip()
        name = inv.get(lab, lab)  # if numeric, map to name
        if name == "NORMAL": 
            continue
        two = label_map_3to2[name]
        # image path relative to split
        src_img = Path(src_split, r["image"])
        dst_img = Path(dst_split, r["image"])
        copy_or_link(src_img, dst_img, copy_imgs)
        out_rows.append({"image": r["image"], "label": str(two), "patient_id": r["patient_id"]})
        kept[name] += 1

    write_csv(out_rows, Path(dst_split, "labels.csv"))
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of 3-class dataset (cbis_cls3)")
    ap.add_argument("--dst", required=True, help="output root for 2-class (cbis_cls2)")
    ap.add_argument("--copy", action="store_true", help="copy images (default: hardlink/copy fallback)")
    args = ap.parse_args()

    counts = {"train":{}, "val":{}, "test":{}}
    for split in ["train","val","test"]:
        kept = filter_split(Path(args.src, split), Path(args.dst, split), args.copy)
        counts[split] = kept

    # write mapping and counts
    with open(Path(args.dst, "class_mapping.json"), "w") as f:
        json.dump({"BENIGN":0,"MALIGNANT":1}, f, indent=2)
    with open(Path(args.dst, "class_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

    print("[DONE] Wrote 2-class dataset at:", args.dst)
    print("Counts:", json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
