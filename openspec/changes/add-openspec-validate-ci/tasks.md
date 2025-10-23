## 1. Implementation
- [ ] 1.1 Create CI workflow that runs `openspec validate` for PRs (target: `.github/workflows/openspec-validate.yml`)
- [ ] 1.2 Add conditional job/path filters so the workflow runs when files under `openspec/` change
- [ ] 1.3 Add a small wrapper script (optional) that detects change-id in the PR and runs `openspec validate <change-id> --strict` or `openspec validate --strict`
- [ ] 1.4 Add tests or dry-run to ensure the workflow can execute in the chosen CI environment
- [ ] 1.5 Update `openspec/project.md` with the CI requirement and usage instructions

## 2. Documentation
- [ ] 2.1 Add a short section to CONTRIBUTING.md or `openspec/project.md` describing the CI check and how to fix common validation errors

## 3. Rollout
- [ ] 3.1 Merge workflow into `main` and monitor PRs for validation failures
- [ ] 3.2 Offer a small PR template or checklist reminding authors to `openspec validate` locally
