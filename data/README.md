# Dataset setup

This directory is intentionally empty in Git. Keep datasets local unless their licenses explicitly permit redistribution.

Expected structure:

```
data/
  train/ai_generated/
  train/real/
  val/ai_generated/
  val/real/
  test/ai_generated/
  test/real/
```

## Dataset protocol

1. Record each image's dataset/source, generator (if known), original identifier, license and split in a CSV manifest.
2. Deduplicate exact and near-duplicate images before splitting.
3. Split by source and generator where possible. For a cross-generator experiment, hold out entire generator families from train/validation and use them only in test.
4. Balance classes or use class weighting if the distribution is strongly imbalanced.
5. Keep the final test split locked until model selection is complete.

Do not treat missing EXIF as an AI label. Avoid mixing edited derivatives of the same source image across splits.
