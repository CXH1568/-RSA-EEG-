# Execution ledger — docs/IMPLEMENTATION_PLAN.md

User authorized sequential Tasks 1–5. No experiments or figure rendering may run.

Ruling: use the explicitly authorized new release directory as the isolated workspace; do not create a Git worktree, commit, or write skill bookkeeping outside it. Cost: no commit-based rollback; original archive remains available and read-only.

Ruling: user-limited syntax/import/help/file checks override the skill's general numerical/full-suite testing suggestions. These checks do not establish numerical equivalence.

Pre-flight interfaces: Task 2 supplies workflow.run(settings) and runtime configuration to Task 3. Task 3 supplies import-safe entrypoints for Task 4. Task 5 requires all Task 4 checks to pass before ZIP creation.

Task 1: complete. All ten submission scripts read completely. Archive baseline: 2,326 file hashes stored in tool session; original ZIP SHA256 A61B88C3DE4ED65D9FBEE13EB66C44004BA4E18F3FE91F44E18D67EA4429073F. Entry-file check exit 1 (expected RED: run_primary absent). No experiment ran.

Task 2: complete. Workflow check RED (missing workflow) -> GREEN (10/10 syntax and import, exit 0). Twenty original non-path helper function ASTs match. No run function called. New files: src/runtime.py, two package initializers, ten workflow files, configs/paths.ini. Scientific correctness and numeric equivalence remain unverified.

Task 3: complete. Entry check RED -> GREEN. Four help commands exit 0; all 20 Python files parse/import; file allowlist passes. Files: four scripts, src/cli.py, src/visualization.py. No workflow or renderer called.

Task 4: BLOCKED. README, license, environment files and ignore rules created. Twenty release Python files parse/import and four CLI help commands exit 0. A separate real scientific dependency import probe did not finish after 207 seconds and was terminated; process exit 1, no PASS marker. Exact dependency responsible is NOT_IDENTIFIED. Per user instruction, stop immediately: no Task 5, no ZIP.

Task 4 safety scan: git grep found no credential/private-server candidates. Broad path search produced three false positives (public CLIP HTTPS URL and two newline-escape strings), no local absolute path. Archive read-only recheck: 2,326 files, zero changed/missing files, ZIP hash unchanged. Validation status recorded in docs/VALIDATION.md.

Task 5: NOT_STARTED. No archive generation or GitHub upload performed.

Ruling: preserve the historical numerical algorithms and disclose the known alignment/BH limitations rather than silently repair them; this is code packaging, not scientific validation. Cost: source methods need independent scientific review before using inferential outputs.

Source review findings: top-level computation in all scripts; personal path assignments; missing-image fallback; catch-and-continue participant failures; legacy retained-event truncation; historical BH implementation needs scientific review. Preserve numerical algorithms with explicit limitations, do not certify or repair these scientific issues during packaging. Input-error behavior can be made fail-fast as authorized.

## Authorized continuation: code-only acceptance

The user removed full scientific-dependency import from release gates. The BLOCKED record above is historical and retained. Scientific dependency import is now NOT_VERIFIED and will not be investigated further. Environment completeness and the cause of the previous timeout are not established. No experiment, model loading, numerical calculation or rendering is authorized.

Task 4 resumed: README, VALIDATION.md and IMPLEMENTATION_PLAN.md updated to separate code-package acceptance from scientific/environment verification. Fresh limited checks and original-archive hash verification precede Task 5.

Task 4: complete under amended user authorization. Fresh limited check: 20 Python files and 4/4 help commands PASS; file allowlist PASS; no secret/private-server candidates; no real local absolute-path hits. Archive hashes: 2,326 files, zero changes/missing files, ZIP unchanged. No full dependency import repeated.

Task 5: packaging authorized after Task 4 PASS. Only public code/doc/config/license files are selected; local IMPLEMENTATION_PLAN.md and PROGRESS.md remain here but are excluded from the ZIP. The packaging command must use exclusive creation, verify CRC and every entry hash, and report its final status. No upload, experiment, rendering or numerical verification.

Task 5: complete. Exclusive ZIP creation exit 0; 28 files, 77,797 bytes. CRC, exact entry list, all entry SHA256 comparisons, and unchanged release-file hashes during packaging PASS. ZIP SHA256: f251d6d16147cd8e55f958158e0af4875a8dfe53441ecaeaacbda93ca657f27c. This excluded local ledger update does not change the delivered ZIP. CODE_PACKAGE_ONLY=YES; SCIENTIFIC_DEPENDENCY_IMPORT=NOT_VERIFIED; EXPERIMENTS=NOT_RUN; NUMERICAL_RESULTS=NOT_VERIFIED. STOP.
