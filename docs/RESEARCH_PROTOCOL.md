# VERIFAI research protocol

## Research question
How well do fine-tuned ImageNet-pretrained CNNs distinguish real photographs from AI-generated images, and how stable are their predictions across unseen generators and common image transformations?

## Models
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

For each model, document the torchvision weight enum/version, preprocessing, random seed, optimizer, learning rate, batch size, epochs, freeze schedule, GPU, training duration and checkpoint hash. Use the same source-aware data partitions for fair comparison.

## Splits and leakage control
A random image-level split can overestimate generalization when images share prompts, source photos, generator pipelines or post-processing. Maintain a manifest and group related images. Reserve at least one generator family for a cross-generator test if dataset coverage permits. Keep the final test split untouched during tuning.

## Metrics
Report accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC, confusion matrix, false-positive rate and inference latency. Include class counts and confidence intervals (bootstrap recommended). Report metrics separately by generator and transformation.

## Explainability
Grad-CAM is a visualization of model activation/attention, not a ground-truth localization map. Evaluate explanations qualitatively with examples and, if pixel-level artifact annotations are available, quantitatively against those annotations. Do not call heatmap regions "fake pixels" without validated localization evidence.

## Robustness
Evaluate JPEG compression, resizing, blur and crop using the same locked test images. Report the performance change relative to original images. Add screenshot and social-media re-encoding tests only when their transformations are reproducible and documented.

## Forensic side channels
Treat EXIF and C2PA as separate provenance signals. Missing metadata is common and must not be used as proof of AI generation. C2PA requires cryptographic manifest validation against trust policy; a simple metadata string search is not verification.

## Reporting
Do not advertise a target accuracy before experiments. Publish exact dataset versions/licenses, split manifests (where permitted), hyperparameters, model checkpoint hashes, limitations and failure examples. Use the phrase "likely AI-generated" rather than "fake".
