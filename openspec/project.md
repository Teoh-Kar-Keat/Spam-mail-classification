## Project Context

### Purpose
This repository follows an OpenSpec-driven development process: specifications (truth) live in `openspec/specs/`, and proposed changes live in `openspec/changes/`. The project's immediate goal is to manage and review change proposals using the OpenSpec workflow, keep requirements executable and reviewable, and ensure spec validation runs automatically on contributions.

### Tech Stack
- OpenSpec (CLI-driven spec system using Markdown)
- Git (feature-branch PR workflow)
- GitHub (recommended) or other Git hosting for PRs and CI
- Ripgrep (`rg`) for local full-text search of specs (optional)
- CI: GitHub Actions (recommended) to run `openspec validate`
- (Languages & frameworks used by code in the repo should be documented here — currently unspecified)

### Project Conventions

#### Code Style
- Follow language-appropriate linters/formatters (Prettier/ESLint for JS/TS, Black/isort for Python, rustfmt for Rust, etc.).
- Commit messages: short summary in imperative mood. Use a prefix when helpful, e.g. `spec:`, `feat:`, `fix:`.
- Files created by proposals should include a clear one-line header describing purpose.

#### Architecture Patterns
- Single-capability-per-spec: each capability in `openspec/specs/<capability>/spec.md` should express a narrowly-scoped behavior.
- Prefer simple, well-tested implementations first. Split into modules/services only when empirical performance or complexity needs justify it.

#### Testing Strategy
- Unit tests for code-level behavior using the repo's language test framework.
- Spec validation: every change proposal MUST include spec deltas in `openspec/changes/<change-id>/specs/`. The CI MUST run `openspec validate <change-id> --strict` (or `openspec validate --strict` for repo-wide checks) for PRs.
- PRs must pass both code tests and `openspec validate` before merge.

#### Git Workflow
- Use short-lived feature branches: `git checkout -b feat/<summary>` or `change/<change-id>`.
- Create an OpenSpec change proposal for new features, breaking changes, or architecture updates using verb-led change IDs (kebab-case, e.g. `add-two-factor-auth`).
- PR must reference the change ID and include the proposal files under `openspec/changes/<change-id>/`.

### Domain Context
No domain-specific business rules are present in the repository currently. Add key domain facts here (actors, typical flows, data sensitivity, SLAs) so AI assistants and contributors can write accurate specs.

### Important Constraints
- This workspace has been initialized on Windows (PowerShell); CI configurations and developer scripts should work cross-platform where possible.
- Specs are the source-of-truth. Any behavioral change that affects users or APIs should go through a change proposal.

### External Dependencies
- `openspec` CLI (required for validation and tooling)
- `rg` (ripgrep) recommended for fast spec searches
- CI runners (GitHub Actions or equivalent) for automated validation

### Assumptions
- I assumed the repository's primary coordination mechanism will be OpenSpec + GitHub/Git; if you use GitLab or another provider we can adapt CI examples accordingly.
- The code language(s) are not specified; fill in language/tooling details (linters, test runner names, package managers) when available.

Please review and update domain-specific sections and any language-specific tooling you want enforced in CI.
