## Why
Pull Requests that change specs or introduce new proposals must be validated automatically to avoid format or parsing errors and to ensure proposals meet OpenSpec requirements. Adding CI validation will enforce correctness early and speed up reviews.

## What Changes
- Add a CI job that runs `openspec validate <change-id> --strict` for changes that add or modify files under `openspec/changes/`.
- Add a repository-level CI check that runs `openspec validate --strict` when no change-id is present (full validation), as an optional job.
- Document the CI requirement in `openspec/project.md`.

**BREAKING**: None — this is additive and enforces validation, but may cause PRs that previously passed to fail until their proposals comply.

## Impact
- Affected specs: none (this only enforces validation of changes)
- Affected code: CI configuration files (e.g., `.github/workflows/openspec-validate.yml`) and developer documentation

## Migration
- Create CI workflow file and update README/`openspec/project.md` to document expected checks.
- Communicate to contributors that PRs touching `openspec/` will trigger validation and must pass before merging.

## Rollback
- Remove or disable the CI workflow file to roll back enforcement.
