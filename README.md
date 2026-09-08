# EKG ST-Elevation (Ischemia) Classifier: CNN vs. Transformer

Can a neural network learn to spot the classic warning sign of a heart attack — ST elevation — directly from raw EKG signal? This project builds and compares two deep learning models (a CNN and a Transformer) trained to detect ST-elevation episodes in the [European ST-T Database](https://physionet.org/content/edb/1.0.0/), a public PhysioNet dataset of two-hour, dual-lead EKG recordings with cardiologist-annotated ischemic episodes.

This is the fifth in an ongoing series of applied AI/ML projects in medicine, following prior work on brain hemorrhage classification, lymph node metastasis detection (PCam), DNA methylation-based brain tumor classification, and CNN-based arrhythmia detection.

---

## Summary (non-technical)

EKGs (electrocardiograms) are one of the fastest, most widely used tools for diagnosing a suspected heart attack. One of the clearest warning signs on an EKG is **ST elevation** — a shift in a specific part of the waveform that can indicate reduced blood flow to the heart muscle.

This project trains two different AI models — a CNN and a Transformer — to recognize that pattern automatically from raw EKG signal, and compares how well each one does.

**Key findings:**
- Both models learned to detect ST elevation meaningfully better than random guessing, but neither is close to clinical-grade accuracy.
- The **Transformer outperformed the CNN** on every metric (68% vs. 63% accuracy, 0.74 vs. 0.70 AUC) — a modest but consistent edge, despite this being a relatively small dataset (the kind of setting where Transformers don't always have an advantage).
- A key early bug — patient-specific baseline voltage differences confounding the model — was diagnosed and fixed via per-record signal normalization, improving both models substantially.
- Error analysis on missed cases (false negatives) found no single cause: some were low-amplitude/subtle elevation, others were affected by signal noise or baseline drift — patterns that closely mirror published findings on why *commercial* STEMI-detection algorithms also produce false negatives.

See the full write-up on [LinkedIn](#) *(link here)* for the accessible version with figures.

---

## Technical appendix

### Dataset
- **Source:** [European ST-T Database (EDB)](https://physionet.org/content/edb/1.0.0/), 90 records, two leads (V4/MLIII depending on record), 250 Hz sampling rate, 2-hour duration each.
- **Labels:** ST-elevation episodes are encoded in the `.atr` annotation files via `'s'`-symbol comment annotations with `aux_note` strings like `(ST1+`, `AST1+200`, `ST1+)` — marking onset, peak, and offset of each episode, along with magnitude in µV. These required custom parsing (see `parse_st_episodes` in the notebook) since they are not part of the standard beat-annotation symbol set.
- **Total episodes found:** 364 (183 lead ST0, 181 lead ST1), close to PhysioNet's documented count of 367.

### Windowing & labeling
- Each record was split into 4-second (1000-sample) windows.
- **Positive windows:** sampled from inside each annotated episode's onset→offset span.
- **Negative windows:** randomly sampled from elsewhere in the same record, with a 30-second exclusion buffer around any episode to avoid ambiguous near-onset/offset samples.

### Critical preprocessing fix: per-record baseline normalization
Initial models performed *worse than random chance* (49% accuracy) on held-out patients. Root cause: absolute EKG signal levels vary drastically by record (patient-specific baseline offsets from electrode placement/recording gain, ranging roughly from -5.3 to +9.0 across records in this dataset), unrelated to actual cardiac status. The model was learning to associate absolute voltage levels with the ischemic label — a pattern that doesn't generalize across patients.

**Fix:** subtract each record's own whole-signal median before windowing, centering every record's baseline near zero. This lets the model learn relative deviation from a patient's own baseline (which is what ST elevation clinically is) rather than absolute voltage. This single change improved CNN test accuracy from 49% to ~63-66%.

### Train/test split
Records were split at the **subject level**, not the record or window level. Several EDB records are documented (per PhysioNet) to originate from the same underlying patient (e.g., e0118–e0122); these were grouped so no patient appears in both train and test. Final split: 68 train records / 17 test records (76 unique subjects, 80/20 split).

### Models

**CNN** — 3-layer 1D convolutional network (32→64→128 filters) with max pooling, global average pooling, and a dense classification head. ~43.6K parameters.

**Transformer** — raw signal is first downsampled via two strided Conv1D layers (1000 → 62 timesteps) before two Transformer encoder blocks (multi-head self-attention + feed-forward), followed by global average pooling and a dense head. ~98.4K parameters. Downsampling before attention is standard practice for long raw time series, where full self-attention over every raw sample is computationally wasteful and rarely helps for predominantly local waveform features.

Both models trained with class-weighted binary cross-entropy (to address the ~62/38 positive/negative class imbalance) and early stopping on validation AUC.

### Results (held-out test set)

| Metric | CNN | Transformer |
|---|---|---|
| Accuracy | 63% | 68% |
| Precision (Ischemic) | 0.75 | 0.79 |
| Recall (Ischemic) | 0.61 | 0.66 |
| AUC | 0.697 | 0.739 |

### Error analysis
Manual and quantitative review of false negatives (missed true ST-elevation windows) found three distinct contributing patterns, not one:
1. **Low amplitude** — some missed windows showed measurably smaller signal amplitude range than correctly-caught positives (quantified via `amplitude_range`, `std` on flattened window arrays).
2. **Signal noise/artifact** — some false negatives showed dense high-frequency noise consistent with muscle artifact or poor electrode contact.
3. **Baseline drift** — some false negatives showed slow, large-scale signal drift unrelated to the cardiac cycle, potentially confounding the elevation signal.

This pattern closely parallels published findings on real-world commercial STEMI-detection software: Bosson et al. (2017, *Prehospital Emergency Care*) found that false positives in commercial algorithms were primarily driven by ECG artifact, while false negatives were tied to ST-segment/T-wave ratios that fell below the algorithm's detection threshold — mirroring the amplitude- and noise-driven failure modes found here.

### Known limitations
- Single-lead analysis (V4/MLIII depending on record); real clinical EKGs use 12 leads.
- Small dataset by deep-learning standards (~52K windows from 90 patients).
- EDB is an older research database; not validated against modern clinical ground truth (e.g., angiography-confirmed STEMI).
- Window-level classification, not full-record or patient-level clinical diagnosis.
- This is a research/educational comparison, not a validated or clinically-usable diagnostic tool.

### Repository structure
```
├── ekg-stt-ischemia-classifier-cnn-transformer.ipynb   # Full analysis notebook
├── stemi_diagram.png                                    # Original P/QRS/T + ST elevation diagram
├── requirements.txt                                     # Python dependencies
└── README.md
```

### Requirements
See `requirements.txt`. Core dependencies: `wfdb`, `tensorflow`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

### References
- Bosson, N. et al. (2017). Causes of prehospital misinterpretations of ST elevation myocardial infarction. *Prehospital Emergency Care*, 21(3), 283-290.
- Mishra, A., Mishra, S., & Mishra, J.P. (2019). Cardiologist or Computer: Who Can Read EKG Better? *Cardiac*, 1(1).
- Taddei, A. et al. The European ST-T Database. PhysioNet.




<img width="905" height="604" alt="Screenshot 2026-09-07 at 16-22-16 Brain CT Hemorrhage Assessment" src="https://github.com/user-attachments/assets/648539b9-0220-4e4e-9e80-69a14ab742b2" />

