# Protocols and provenance records

| File | What it is | Recorded |
| --- | --- | --- |
| `confirmatory/CONFIRMATORY_PROTOCOL.md` | Pre-registered protocol for the confirmatory study (H1–H3) | frozen 2026-11-04 |
| `confirmatory/CONFIRMATORY_PROTOCOL.sha256` | SHA-256 of the protocol text and the time the hash was taken | Wed Nov 4 16:32:06 UTC 2026 |
| `confirmatory/code_at_test_run.sha256` | SHA-256 of `formal_si/confirm.py` and `formal_si/translit.py` at the single test evaluation | Wed Nov 4 17:14:11 UTC 2026 |
| `exploratory/POST_CONFIRMATORY_PROTOCOL_V2.md` | Post-confirmatory, exploratory controls (written after the confirmatory test run) | 2026-11-04 |
| `exploratory/POST_CONFIRMATORY_PROTOCOL_V2.sha256` | SHA-256 of the v2 protocol text and the time the hash was taken | Wed Nov 4 17:57:30 UTC 2026 |

`python scripts/verify_artifacts.py` checks all three records against the files in this repository.

Notes for readers:

- The protocol texts and the code hashes are unchanged copies of the originals. The code hashes refer to paths
  inside the package (`formal_si/...`). In this repository the package lives under `src/`, so the files are
  `src/formal_si/confirm.py` and `src/formal_si/translit.py`. Their contents are unchanged.
- The original `CONFIRMATORY_PROTOCOL.sha256` record named the protocol by an absolute path on the machine that
  ran the study. Here that path is replaced by the bare file name. The hash and the timestamp are unchanged.
- The timestamps come from the system clock of the machine that ran the study. They are not attested by a third
  party, for example a registry or a trusted timestamping service.
- The v2 protocol text says it was written "~18:10 UTC". Its hash record was taken at 17:57:30 UTC. The
  machine-recorded time is the reliable one. Both are after the confirmatory test run (17:14:11 UTC).
- The confirmatory protocol says the encoder revisions were "pinned". The original code loaded each model's
  Hub `main` branch, which resolved to the listed revisions at run time. This release pins them in code as well
  (see `docs/REPRODUCIBILITY.md`).
- The confirmatory protocol mentions `reports/confirm/negation_review_100.csv`. That file contains corpus
  sentences and is not distributed (see `docs/DATASETS.md`). No claim rests on it.
- `POST_CONFIRMATORY_PROTOCOL_V2.sha256` is unchanged and names the file by its original relative path
  (`protocol/POST_CONFIRMATORY_PROTOCOL_V2.md`). `verify_artifacts.py` matches such records by file name.
