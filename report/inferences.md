# Key inferences (Day 11, from the Day 10 grid)

Source: `experiments/results/grid.csv` (7 models x 8 streams, scenarios x 5 seeds, 224 runs),
tables in `report/tables/`, figures `analysis/figures/fig1-5*`. "Baseline" = the paper's informed
ageing MNB with the vocabulary detector (Rebuild-Zero); "enhanced" = the same model + ADWIN on the
prediction error; OR is the main enhanced model, ADWIN-only and AND are ablations.

1. **The baseline reproduces the paper's ordering.** On DS1 and DS2 the accumulative MNB is worst
   (78.3 / 80.3%), fixed ageing helps (80.2 / 82.5%) and informed adaptation is best among the
   paper's models (80.8-81.4 / 83.0%). Adapting pays off after the natural change point (DS1 tail
   ~99-100% for adaptive models vs 77% accumulative). Two differences from the paper: our
   accumulative baseline is ~5 pt stronger, and lambda0 = 0 ("-Zero-") variants beat lambda0 > 0
   ("-Init-") because constant ageing costs 1-2 pt in stable periods (Day 6).

2. **The vocabulary detector is blind to meaning change; ADWIN is not.** When only the meaning
   changes and the words stay the same (S-label-flip-topic), the vocabulary detector misses the
   drift in 2 of 5 runs and detects it ~4,000 instances late otherwise; ADWIN detects it in every
   run after ~120 instances (Fig. 3). On the full label flip, ADWIN reacts after 39 instances vs
   3,999 (a ~100x shorter delay).

3. **Faster detection means faster recovery and higher accuracy where it matters.** On S-label-flip
   the enhanced model recovers to 95% of its pre-drift accuracy in 3,183 instances vs 8,204 for the
   baseline, and gains +2.8 pt accuracy (73.8 +- 0.3 vs 71.0 +- 1.0%, 5 seeds); its minimum
   accuracy after the flip is ~58% vs ~25% for the baseline (Fig. 2).

4. **It also helps on real data.** DS1 81.3% (OR) / 81.7% (AND) vs 80.8% for the same-base
   baseline; DS2 83.7 / 84.0% vs 83.0%. ADWIN flags DS1's natural change point 704 instances after
   it vs 19,224 for the vocabulary detector. Every enhanced variant beats its Rebuild-Zero baseline
   on both real streams.

5. **The two detectors are complementary, which is why OR is the main model.** ADWIN only reacts
   when the error rises, so it ignores pure prior shifts and topic switches; the vocabulary
   detector catches the prior shift (81.6% with it vs 80.3% without). OR keeps both: it detects the
   most drifts (39 / 47 vs 36 / 47 for the baseline and 18 / 47 for ADWIN-only) and never loses
   accuracy against the baseline by more than 0.04 pt on any stream (Fig. 5a). OR >= baseline in
   28 of 32 stream-seed pairs.

6. **The price of OR is false alarms, almost all inherited from the vocabulary detector.**
   OR raises 102 false alarms over the grid vs 70 for the baseline, mostly on DS1 / DS2 whose ground
   truth lists only the designed drift points, so these counts are upper bounds. ADWIN-only and AND
   raise 29-30. Rebuild-on-alarm makes false alarms cheap here: accuracy still goes up.

7. **Virtual drift barely matters for accuracy.** On sudden / gradual / recurring topic switches
   (P(X) changes, P(y|X) does not) every model stays within ~0.7 pt; the class-conditional word
   model transfers across topics. Detecting such drift is informative (vocabulary detector) but not
   needed for accuracy - another argument for an error-based trigger.

8. **Lambda-based strategies depend on the time scale; rebuild does not.** SlowIncreaseFastReset is
   the best baseline on DS1 (81.4%) but collapses on S-label-flip (54.5%): lambda is per hour and a
   scenario spans ~22 synthetic hours, so a higher lambda barely ages the model. Rebuild works on
   both, which is why the enhancement is built on it.

9. **Naive Bayes has a ceiling on conditional meaning change.** A flip limited to one topic cannot
   be learned by MNB (words are scored independently), so every model, even with instant detection,
   tops out at ~64% after it. Detection helps (+0.3-0.9 pt) but the model class limits recovery.

10. **The enhancement is cheap.** Offline throughput 37-46k tweets/s for all adaptive models
    (ADWIN < 1 us per update); in Spark the models take 23-26 ms (baseline) vs 29-33 ms (enhanced)
    per 1k tweets against ~170 ms fixed micro-batch overhead. The pipeline sustains ~2,000 tweets/s
    with sub-second latency at a 1k batch cap and ~5,600 tweets/s with a 5k cap.
