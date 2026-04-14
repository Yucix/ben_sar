import os
import json
import random
from collections import OrderedDict
from tqdm import tqdm

ROOT = "/media/sata/xyx/BigEarthNet/dataset"

TRAIN_CSV = os.path.join(ROOT, "train.csv")
VAL_CSV = os.path.join(ROOT, "val.csv")
TEST_CSV = os.path.join(ROOT, "test.csv")

CLOUD_CSV = os.path.join(ROOT, "patches_with_cloud_and_shadow.csv")
SNOW_CSV = os.path.join(ROOT, "patches_with_seasonal_snow.csv")

OUT_TRAIN = os.path.join(ROOT, "train_10p_clean_622.csv")
OUT_VAL = os.path.join(ROOT, "val_10p_clean_622.csv")
OUT_TEST = os.path.join(ROOT, "test_10p_clean_622.csv")

S1_ROOT = os.path.join(ROOT, "BigEarthNet-S1-v1.0")

SEED = 3407
TARGET_RATIO = 0.10
SPLIT_RATIO = (0.6, 0.2, 0.2)

# 每次多抽一些，减少补抽次数
BUFFER_SIZE = 500

CLASS43 = [
    'Continuous urban fabric',
    'Discontinuous urban fabric',
    'Industrial or commercial units',
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Non-irrigated arable land',
    'Permanently irrigated land',
    'Rice fields',
    'Vineyards',
    'Fruit trees and berry plantations',
    'Olive groves',
    'Pastures',
    'Annual crops associated with permanent crops',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland',
    'Moors and heathland',
    'Sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Sparsely vegetated areas',
    'Burnt areas',
    'Inland marshes',
    'Peatbogs',
    'Salt marshes',
    'Salines',
    'Intertidal flats',
    'Water courses',
    'Water bodies',
    'Coastal lagoons',
    'Estuaries',
    'Sea and ocean',
]
CLASS2IDX_43 = {c: i for i, c in enumerate(CLASS43)}

LABEL_CONVERTER = {
    0: 0, 1: 0, 2: 1,
    11: 2, 12: 2, 13: 2,
    14: 3, 15: 3, 16: 3, 18: 3,
    17: 4,
    19: 5,
    20: 6,
    21: 7,
    22: 8,
    23: 9,
    24: 10,
    25: 11, 31: 11,
    26: 12, 27: 12,
    28: 13,
    29: 14,
    33: 15, 34: 15,
    35: 16, 36: 16,
    38: 17, 39: 17,
    40: 18, 41: 18, 42: 18,
}


def read_pairs(csv_path):
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s2_name, s1_name = line.split(",")
            pairs.append((s2_name, s1_name))
    return pairs


def read_patch_list(csv_path):
    names = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    return names


def write_pairs(csv_path, pairs):
    with open(csv_path, "w", encoding="utf-8") as f:
        for s2_name, s1_name in pairs:
            f.write(f"{s2_name},{s1_name}\n")


def has_valid_19_label(s1_patch_name):
    json_path = os.path.join(S1_ROOT, s1_patch_name, f"{s1_patch_name}_labels_metadata.json")
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    labels43 = meta.get("labels", [])
    idx43 = [CLASS2IDX_43[x] for x in labels43 if x in CLASS2IDX_43]
    idx19 = [LABEL_CONVERTER[i] for i in idx43 if i in LABEL_CONVERTER]
    return len(idx19) > 0


def main():
    rng = random.Random(SEED)

    print("Reading original splits...")
    train_pairs = read_pairs(TRAIN_CSV)
    val_pairs = read_pairs(VAL_CSV)
    test_pairs = read_pairs(TEST_CSV)

    original_total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"Original total = {original_total}")

    all_pairs = train_pairs + val_pairs + test_pairs

    # 去重
    dedup = list(OrderedDict(((s2, s1), None) for s2, s1 in all_pairs).keys())
    print(f"After dedup = {len(dedup)}")

    print("Reading blacklist files...")
    cloud_set = read_patch_list(CLOUD_CSV)
    snow_set = read_patch_list(SNOW_CSV)
    print(f"Cloud/shadow patches = {len(cloud_set)}")
    print(f"Seasonal snow patches = {len(snow_set)}")

    # 先只做快速过滤
    candidate_pairs = []
    removed_cloud = 0
    removed_snow = 0

    for s2_name, s1_name in tqdm(dedup, desc="Filtering cloud/snow", total=len(dedup)):
        if s2_name in cloud_set:
            removed_cloud += 1
            continue
        if s2_name in snow_set:
            removed_snow += 1
            continue
        candidate_pairs.append((s2_name, s1_name))

    print(f"After cloud/snow filtering = {len(candidate_pairs)}")
    print(f"Removed cloud/shadow = {removed_cloud}")
    print(f"Removed seasonal snow = {removed_snow}")

    target_total = int(round(original_total * TARGET_RATIO))
    print(f"Target sample count (10% of original) = {target_total}")

    if len(candidate_pairs) < target_total:
        raise ValueError(
            f"过滤后候选池只有 {len(candidate_pairs)} 个样本，不足以抽取 {target_total} 个。"
        )

    # 打乱候选池，然后顺序扫描，避免反复随机抽样
    rng.shuffle(candidate_pairs)

    selected_pairs = []
    invalid19_count = 0
    checked_count = 0

    # 为了避免重复检查同一个样本，加一个简单缓存
    valid_cache = {}

    pbar = tqdm(total=target_total, desc="Selecting valid 19-label samples")

    for s2_name, s1_name in candidate_pairs:
        if len(selected_pairs) >= target_total:
            break

        checked_count += 1

        if s1_name in valid_cache:
            is_valid = valid_cache[s1_name]
        else:
            is_valid = has_valid_19_label(s1_name)
            valid_cache[s1_name] = is_valid

        if is_valid:
            selected_pairs.append((s2_name, s1_name))
            pbar.update(1)
        else:
            invalid19_count += 1

    pbar.close()

    if len(selected_pairs) < target_total:
        raise ValueError(
            f"筛选后只得到 {len(selected_pairs)} 个有效样本，少于目标 {target_total}。"
        )

    print(f"Checked samples for 19-label validity = {checked_count}")
    print(f"Removed invalid 19-label in sampled process = {invalid19_count}")
    print(f"Final selected total = {len(selected_pairs)}")

    # 最后再打乱一次，避免因为候选池扫描顺序带来偏置
    rng.shuffle(selected_pairs)

    n_train = int(target_total * SPLIT_RATIO[0])
    n_val = int(target_total * SPLIT_RATIO[1])
    n_test = target_total - n_train - n_val

    train_out = selected_pairs[:n_train]
    val_out = selected_pairs[n_train:n_train + n_val]
    test_out = selected_pairs[n_train + n_val:]

    print("\nFinal split:")
    print(f"train = {len(train_out)}")
    print(f"val   = {len(val_out)}")
    print(f"test  = {len(test_out)}")
    print(f"total = {len(train_out) + len(val_out) + len(test_out)}")

    print("Writing output csv files...")
    write_pairs(OUT_TRAIN, train_out)
    write_pairs(OUT_VAL, val_out)
    write_pairs(OUT_TEST, test_out)

    print("\nSaved to:")
    print(OUT_TRAIN)
    print(OUT_VAL)
    print(OUT_TEST)


if __name__ == "__main__":
    main()