# Limited release validation — code-only scope

Date: 2026-09-23. This is a code-packaging check, not a scientific audit or experiment result.

| Check | Observed result |
| --- | --- |
| Python syntax | PASS: 20 Python files |
| Release module import | PASS: workflow modules and entrypoints; no run/render function called |
| Four CLI help commands | PASS: 4/4 exit 0 |
| Ten workflow import guards | PASS: no top-level scientific execution |
| Preserved original helper-function ASTs | PASS: 20 non-path helpers unchanged; not a numerical-equivalence claim |
| File-extension/directory allowlist | PASS: no datasets, results, binary assets or caches |
| git grep credential/private-server scan | No candidates (exit 1 means no match) |
| git grep broad absolute-path scan | Three false positives reviewed; no local absolute path |
| Actual scientific dependency import | NOT_VERIFIED; not a release gate under the revised user authorization |
| Experiments / inference / group statistics | NOT_RUN |
| Renderer / new scientific figures | NOT_RUN |
| Original archive integrity | 2,326 files checked; no changed/missing files; original ZIP hash unchanged |
| ZIP delivery | Packaged after the current gates pass; delivery size/checksum are reported separately |
| GitHub publication | NOT_PERFORMED |

## Commands run

```bash
python -B docs/check_release.py --stage entries
python -B docs/check_release.py --stage workflows
python -B docs/check_release.py
```

The first entries/workflows checks failed because files were intentionally not implemented yet (RED). After implementation, the complete check exited 0, covering syntax, module import, four help commands and the file allowlist. The installed local analysis interpreter was used; no environment installation was attempted.

`git grep --no-index --no-exclude-standard` scanned release text for credential indicators, private-server patterns and local paths. The broad path pattern matched the public OpenAI CLIP source URL in requirements and two escaped newline strings in the full-cohort workflow. These are not private filesystem dependencies. No values resembling credentials were printed or stored in this report.

## Historical dependency probe (not a current release gate)

The separate dependency check attempted imports of Pillow, NumPy, pandas, MNE, torch, torchvision, OpenAI CLIP, SciPy, scikit-learn and Matplotlib, including the actual Ridge/PCA/statistical/model APIs. It did not reach its completion marker within 207 seconds. The process was identified by its process ID and start time and terminated. No analysis function, model constructor, data loader or renderer was called.

The precise import that stalled is unknown: that composite probe did not print per-dependency progress. This is not evidence of numerical failure or proof that a particular library is broken. The full research environment is not established as installed and validated for this release. No environment-level dependency verification is undertaken in the current code-only scope. The historical timeout is not attributed to missing packages without evidence; no further diagnosis is performed.

The user explicitly removed full scientific-dependency import from ZIP acceptance. Current gates are syntax, CLI help, file allowlist, no bundled data/results, no detected sensitive information/local absolute paths, and unchanged original archive.

## Final code-package gates

The limited checker was rerun after the documentation amendment: all 20 Python files parsed/imported, four CLI help commands passed, and the file allowlist passed. No science dependency probe was restarted. Credential/private-server scanning again returned no matches; the same three broad-path false positives were reviewed. A fresh comparison of all 2,326 original archive files and the original ZIP found no changes.

Task 4 is complete under the amended scope. The delivered ZIP contains the public code, configuration, environment declarations, license, README, code-origin notes, this report and the limited checker. Local implementation-plan and progress-ledger documents are deliberately excluded, not deleted. Packaging verifies the file list, ZIP CRC and each entry's bytes against the release files. GITHUB_READY denotes code-package readiness only, not environment validation, scientific replication or an actual GitHub upload.

SCIENTIFIC_DEPENDENCY_IMPORT = NOT_VERIFIED

EXPERIMENTS = NOT_RUN

NUMERICAL_RESULTS = NOT_VERIFIED

CODE_PACKAGE_ONLY = YES
