## ADDED Requirements

### Requirement: CI Validation for OpenSpec changes
The repository SHALL run `openspec validate` in CI for pull requests that add or modify files under the `openspec/` directory. Validation SHOULD be run in `--strict` mode to ensure strict compliance with spec formatting and scenario requirements.

#### Scenario: PR touching openspec/ triggers change-id validation
- **WHEN** a Pull Request modifies files under `openspec/changes/` and includes a new `changes/<change-id>/` directory
- **THEN** the CI job SHALL run `openspec validate <change-id> --strict` and fail if validation reports errors

#### Scenario: PR touches openspec/ without change-id
- **WHEN** a Pull Request modifies files under `openspec/` but does not include a single change-id directory
- **THEN** the CI job SHALL run `openspec validate --strict` (full repo validation) and fail if validation reports errors

#### Scenario: Non-openspec PRs skip validation
- **WHEN** a Pull Request does not modify files under `openspec/`
- **THEN** the `openspec validate` job SHALL be skipped to reduce CI time
