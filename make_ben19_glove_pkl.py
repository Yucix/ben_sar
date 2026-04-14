import os
import pickle
import numpy as np

GLOVE_TXT = "/media/sata/xyx/BigEarthNet/dataset/embeddings/glove.6B.300d.txt"
OUT_PKL = "/media/sata/xyx/BigEarthNet/dataset/embeddings/bigearthnet19_glove_word2vec.pkl"

BEN19_CLASSES = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters',
]

def load_glove(glove_path):
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

def phrase_to_vec(phrase, glove, dim=300):
    # 小写 + 去掉逗号
    phrase = phrase.lower().replace(",", "")
    words = phrase.split()

    vecs = [glove[w] for w in words if w in glove]

    if len(vecs) == 0:
        print(f"[WARN] no glove words found for: {phrase}")
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vecs, axis=0)

def main():
    glove = load_glove(GLOVE_TXT)
    embs = []

    for cls in BEN19_CLASSES:
        vec = phrase_to_vec(cls, glove, dim=300)
        embs.append(vec)
        print(f"done: {cls}")

    embs = np.stack(embs, axis=0)   # [19, 300]

    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(embs, f)

    print(f"saved to: {OUT_PKL}")
    print(f"shape: {embs.shape}")

if __name__ == "__main__":
    main()