# Technical interview preparation

The CrowdStrike AI Research Launch Intern role, requisition R30599, emphasizes identifying gaps, implementing algorithms from scratch, examining mathematical assumptions, evaluating new methods and writing without GenAI assistance. This project offers examples involving calibration under drift, backpropagation derived and checked with finite differences, and a case where threshold adaptation hurts ranking or recall. The repository is AI-assisted. Use the questions to practice independent derivations and write your own sample for any application that requests unaided writing.

1. **Why score before updating?** Otherwise the model has already seen the batch it is being assessed on. The resulting alert metrics would describe a less realistic online setting.
2. **Why hold out two sets after model fitting?** One fits the rank mapping and another independently sets an alert threshold. A calibration sample cannot be compared fairly to empirical ranks if it helped fit the underlying model.
3. **How did you compare the policies at the same detection level?** On pooled test scores, compute the first attainable ROC point whose recall is at least 60% and compare its FPR. This descriptive analysis uses labels and accompanies the actual deployed alert decision.
4. **How does a change in benign behavior create alerts?** The fixed score distribution and cutoff come from warmup. A benign shift can move normal flows into high-score regions, so a fixed cutoff has a different FPR after the shift.
5. **What drift does the gate miss?** A change limited to a minority of features or samples can leave the median shifts small. A burst that does affect enough features can also trigger the gate in the absence of a benign shift.
6. **Why trim by prior model scores?** Large prior scores can flag questionable training candidates before retraining. Under benign drift, those same scores may reject examples the model needs to learn. Report both admitted contamination and detection outcomes.
7. **How did you derive the VAE gradient?** For z = μ + exp(log σ²/2)·ε, the reconstruction gradient flows into μ directly and into log σ² through exp(log σ²/2)·ε/2. Add the derivatives of the Gaussian KL term and check with fixed noise during finite differences.
8. **What does rank pooling assume?** Scores reflect relative standing against their reference populations. Reference contamination, small calibration windows and shifts in class mix can make successive ranks imperfectly comparable.
9. **Why can an empirical percentile over a mixed batch lower recall?** If several anomalies occupy the high tail, the threshold can move past their scores. A lower false-positive rate may coincide with less attack detection.
10. **What does the public benchmark establish?** UNSW-NB15 measures detector behavior on captured cyber-range features with a constructed partition replay. A timestamped independent dataset would be needed to measure generalization to natural temporal drift.

## Evidence-based résumé draft

Use these as drafts and adjust the verbs to reflect the implementation, reproduction, and analysis you personally performed.

- Evaluated six unsupervised detectors and seven streaming adaptation policies through 1,100 paired replay runs, separating matched-recall ranking comparisons from deployed alert decisions.
- Measured a 35.6-point reduction in false positives at 60% recall for a drift-gated Isolation Forest under synthetic sudden drift, alongside an 8.0-point increase on the UNSW-NB15 replay.
- Verified manually derived NumPy gradients for two autoencoders using finite differences and built an offline dashboard for investigating per-batch alerts and drift.
