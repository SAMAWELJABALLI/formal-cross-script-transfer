# Confirmatory protocol: human vs synthetic transliteration pairs for cross-script invariance

Frozen 2026-11-04 before (a) any test-split string was encoded or read, and (b) the dev results of the
stronger-transliterator control and of the learning curve were computed. Any deviation is reported in the
paper as a deviation. The test split is evaluated ONCE with the code at the commit/hash recorded below.

## Fixed elements (carried over from the pilot / H-C, chosen on dev)
- Encoders (frozen, CPU, revisions pinned): intfloat/multilingual-e5-small 614241f (mean pool, "query: " prefix),
  sentence-transformers/LaBSE 836121a (CLS of last hidden state, no dense head), FacebookAI/xlm-roberta-base
  e73636d (mean pool). L2-normalised, max 128 tokens.
- Text preprocessing: emoji stripped in every source (formal_si.text.normalize.strip_emoji).
- Tasks (train Arabic script -> evaluate human Arabizi = "cross-script"): NArabizi sentiment (Algerian),
  TSAC sentiment (Tunisian), MYC sentiment (Moroccan), NArabizi topic. 4 tasks x 3 encoders = 12 cells.
- Classifier: logistic regression, C in {0.1, 1, 10} chosen on IN-SCRIPT DEV macro-F1; 5 training resamples
  (90% subsamples, seeds from numpy default_rng(0) as in formal_si.run); train split only.
- Invariance removal: h = (I - U U^T) x, U = top-k eigenvectors of the pair-difference scatter
  S_inv = (1/n) sum (a_i - b_i)(a_i - b_i)^T, k = 16 (fixed on dev in H-C before H-C results were seen).
- Pairs: the 998 NArabizi TRAIN sentence groups that have both a human Arabic-script and a human Arabizi version.

## Systems evaluated on test
- S0 raw: frozen embedding, no removal.
- S1 human pairs: removal with S_inv from (human Arabic, human Arabizi) pairs.
- S2 synthetic-rule pairs: same 998 Arabizi sentences, Arabic side = rule deromaniser (formal_si.text.normalize.deromanize).
- S3 synthetic-dictionary pairs (stronger-transliterator control): same 998 Arabizi sentences, Arabic side =
  data-driven token transliterator (formal_si/translit.py): token dictionary learned from (i) TArC train
  Arabizi->Arabic-script tokens of the code-switched layer and (ii) token-count-aligned NArabizi train pairs,
  CROSS-FITTED in 2 folds by group so no sentence is transliterated by a dictionary that saw it; rule fallback for
  unknown tokens.
- S4 derom-rule encode: rule-deromanised input, no removal.  S4b derom-dict encode: dictionary-deromanised input
  (dictionary learned on all NArabizi train + TArC train; no test data).
- S5 char n-gram TF-IDF (char_wb 1-5, LR C=1) on rule-deromanised text; S5b on dict-deromanised text; S6 on raw text.

## Hypotheses (cross-script macro-F1 on test)
- H1: S1 > S0.   H2: S1 > S2.   H3: S1 > S3.
- Secondary (reported, not confirmatory): S1 vs S4, S4b, best of S5/S5b/S6; in-script cost S1 - S0;
  code-switch layer (NArabizi only).
## Statistics
- Per cell: grouped paired bootstrap (resample test sentence groups; 2,000 draws), delta = mean over the 5
  resamples of macro-F1(a) - macro-F1(b); 95% percentile CI; two-sided p. Holm over the 12 cells per hypothesis.
- Aggregate per hypothesis: mean delta over the 12 cells; CI from the average of independent per-cell bootstrap draws.
- A hypothesis is CONFIRMED if the aggregate 95% CI excludes 0 in the predicted direction; per-cell Holm results
  are reported in full. Effects opposite to the prediction are reported as such.
## Learning curve (dev only, not test)
- n in {50, 100, 250, 500, 998} human pairs (3 random draws each), and S2/S3 at the same n; k = min(16, n).
## Linguistic validation
- 100 rule negations sampled from the dev contrast set are exported for native-speaker review
  (reports/confirm/negation_review_100.csv). Until reviewed they are UNVALIDATED; negation results are not used for
  any claim in the paper.
## Not done (by decision): no FORMAL-SI architecture work, no fine-tuning, no GPU.
