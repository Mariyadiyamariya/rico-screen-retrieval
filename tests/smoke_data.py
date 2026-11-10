"""Synthetic stand-ins used only by the offline smoke test (SMOKE=1).

They let the whole notebook run end to end without internet or a GPU: a tiny fake
RICO-like dataset with JPEG screenshots, a word-level BERT tokenizer and randomly
initialised small BERT / CLIP models. Real runs never import this file.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TYPES = ["login", "settings", "map", "chat", "list", "profile", "welcome", "search"]
CATS = ["Shopping", "Social", "Travel & Local", "Finance", "Weather", "Education"]
WORDS = ["page", "of", "a", "an", "app", "showing", "display", "screen", "with", "options", "the", "in", "for",
         "button", "list", "items", "user", "new", "red", "blue", "green", "pop", "up"] + TYPES + [c.lower() for c in CATS]
VOCAB = Path("cache_smoke/vocab.txt")


def _img(kind, seed):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (216, 384), tuple(int(v) for v in rng.integers(180, 255, 3)))
    d = ImageDraw.Draw(im)
    k = TYPES.index(kind)
    for i in range(k + 2):
        y = 30 + i * 40
        d.rectangle([20, y, 196, y + 25], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    b = io.BytesIO(); im.save(b, format="JPEG"); return {"bytes": b.getvalue(), "path": None}


def make_fake_dataset(n=700, seed=0):
    from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ["screenId", "captions", "app_package_name", "play_store_name", "category", "image"]}
    for i in range(n):
        kind = TYPES[rng.integers(len(TYPES))]; app = int(rng.integers(60)); cat = CATS[app % len(CATS)]
        if i == 7: cat = "000 - 1"                                     # malformed category, like the real data
        caps = [f"{kind} page", f"{kind} page of a {cat.split()[0].lower()} app",
                f"display of {kind} screen with {rng.choice(['red', 'blue', 'green'])} button",
                f"{kind} page", f"screen showing {kind} options for user {rng.choice(WORDS)}"]
        rows["screenId"].append(i); rows["captions"].append(caps); rows["app_package_name"].append(f"com.app{app}")
        rows["play_store_name"].append(f"App {app}"); rows["category"].append(cat); rows["image"].append(_img(kind, i))
    feats = Features({"screenId": Value("int64"), "captions": Sequence(Value("string")), "app_package_name": Value("string"),
                      "play_store_name": Value("string"), "category": Value("string"), "image": HFImage()})
    ds = Dataset.from_dict(rows, features=feats)
    a, b = int(n * 0.7), int(n * 0.8)
    return DatasetDict(train=ds.select(range(a)), val=ds.select(range(a, b)), test=ds.select(range(b, n))), "SMOKE (synthetic)"


def fake_tokenizer():
    from transformers import BertTokenizerFast
    VOCAB.parent.mkdir(exist_ok=True, parents=True)
    words = sorted(set(w for c in CATS for w in c.lower().replace("&", " ").split()) | set(WORDS) | {"000", "1", "-", "&"})
    VOCAB.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
    return BertTokenizerFast(vocab_file=str(VOCAB), do_lower_case=True)


def fake_bert():
    from transformers import BertConfig, BertModel
    return BertModel(BertConfig(vocab_size=200, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=128, max_position_embeddings=64))


def fake_clip():
    from transformers import CLIPConfig, CLIPModel
    cfg = CLIPConfig(text_config=dict(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                      num_attention_heads=2, max_position_embeddings=77),
                     vision_config=dict(image_size=224, patch_size=32, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=2), projection_dim=16)
    return CLIPModel(cfg)


def plant_signal(feats, captions_first, cap_owner, seed=0):
    """Replace random-init features with ones that share a known latent between a screen and its captions,
    so the smoke test can check that every model actually learns and that the metrics move."""
    rng = np.random.default_rng(seed)
    n = len(captions_first)
    kind = np.array([TYPES.index(c.split()[0]) if c.split()[0] in TYPES else 0 for c in captions_first])
    z = np.eye(len(TYPES))[kind] * 2 + rng.normal(0, 1.5, (n, len(TYPES)))        # screen-specific latent
    out = dict(feats)
    for k, v in feats.items():
        if v.ndim != 2: continue
        A = rng.normal(0, 1, (len(TYPES), v.shape[1]))
        if v.shape[0] == n:
            out[k] = (z @ A + rng.normal(0, 0.5, v.shape)).astype(np.float16)
        else:
            zc = z[cap_owner] + rng.normal(0, 1.6, (len(cap_owner), len(TYPES)))
            out[k] = (zc @ A + rng.normal(0, 0.5, v.shape)).astype(np.float16)
    return out


def plant_tokens(tokens, feats, seed=1):
    """Token features carrying the same planted signal as the pooled ones (smoke test only)."""
    rng = np.random.default_rng(seed)
    out = dict(tokens)
    img, txt = feats["resnet50"].astype(np.float32), feats["bert_mean"].astype(np.float32)
    R = tokens["img_regions"].shape[1]; L = tokens["txt_tokens"].shape[1]
    out["img_regions"] = (img[:, None, :] + rng.normal(0, 1.0, (len(img), R, img.shape[1]))).astype(np.float16)
    out["txt_tokens"] = (txt[:, None, :] + rng.normal(0, 1.0, (len(txt), L, txt.shape[1]))).astype(np.float16)
    return out
