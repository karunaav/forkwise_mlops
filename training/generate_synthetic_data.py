#!/usr/bin/env python3
"""
generate_synthetic_data.py — Create synthetic training data for ForkWise.

Run this ONCE on the Chameleon node before starting training.
Creates train/val/test splits in the exact format train.py expects.

Uses realistic substitution pairs (butter↔oil, milk↔cream, etc.)
so the model can learn actual patterns and MRR@3 will be meaningful.
Includes cuisine labels for per-cuisine fairness metrics.

Per Hivansh: train directly from this data now to find best
hyperparameters, then swap --dataset path to the real Recipe1MSubs
bucket path later.

Output files:
  data/processed/train.json  (5000 samples)
  data/processed/val.json    (500 samples)
  data/processed/test.json   (500 samples)
"""

import json
import random
from pathlib import Path

random.seed(42)

# Full ingredient vocabulary
INGREDIENTS = [
    "flour", "egg", "sugar", "butter", "milk", "salt", "pepper", "oil",
    "garlic", "onion", "tomato", "chicken", "beef", "rice", "pasta",
    "cheese", "cream", "lemon", "herbs", "vanilla", "baking_powder",
    "yeast", "water", "vinegar", "honey", "soy_sauce", "ginger",
    "cinnamon", "nutmeg", "paprika", "cumin", "oregano", "basil",
    "thyme", "rosemary", "potato", "carrot", "celery", "mushroom",
    "spinach",
]

# Realistic substitution pairs (bidirectional)
SUBSTITUTION_PAIRS = [
    ("butter", "oil"),
    ("milk", "cream"),
    ("sugar", "honey"),
    ("egg", "baking_powder"),
    ("cream", "milk"),
    ("lemon", "vinegar"),
    ("pasta", "rice"),
    ("chicken", "beef"),
    ("basil", "oregano"),
    ("thyme", "rosemary"),
    ("garlic", "onion"),
    ("cinnamon", "nutmeg"),
    ("cumin", "paprika"),
    ("potato", "carrot"),
    ("spinach", "celery"),
    ("cheese", "cream"),
    ("soy_sauce", "salt"),
    ("ginger", "garlic"),
    ("mushroom", "onion"),
    ("flour", "yeast"),
]

CUISINES = [
    "italian", "indian", "mexican", "chinese", "american",
    "french", "thai", "japanese", "mediterranean", "unknown",
]


def make_samples(n):
    """Generate n training samples with realistic substitutions."""
    samples = []
    for _ in range(n):
        orig, repl = random.choice(SUBSTITUTION_PAIRS)

        # Build a random recipe context containing the original ingredient
        num_ctx = random.randint(3, 8)
        ctx_pool = [i for i in INGREDIENTS if i not in (orig, repl)]
        ctx = random.sample(ctx_pool, min(num_ctx, len(ctx_pool)))
        ctx.append(orig)        # recipe must contain the original
        random.shuffle(ctx)

        samples.append({
            "recipe_id": f"r{random.randint(100000, 999999)}",
            "original": orig,
            "replacement": repl,
            "ingredients": ctx,
            "cuisine": random.choice(CUISINES),
        })
    return samples


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": 5000, "val": 500, "test": 500}

    for split_name, n in splits.items():
        data = make_samples(n)
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split_name}: {n} samples → {path}")

    print(f"\nDone. Data format:")
    print(f"  Fields: recipe_id, original, replacement, ingredients, cuisine")
    print(f"  Vocab size: {len(INGREDIENTS)} ingredients")
    print(f"  Substitution pairs: {len(SUBSTITUTION_PAIRS)}")
    print(f"\nReady for training. Run:")
    print(f"  python training/train.py --config training/config.yaml \\")
    print(f"    --dataset data/processed/train.json \\")
    print(f"    --run_name gismo-emb128-gpu")


if __name__ == "__main__":
    main()
