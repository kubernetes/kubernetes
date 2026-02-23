# Contributing to Kubernetes

Welcome to the **Kubernetes** repository! This document provides the essential information you need to get started contributing code, documentation, tests, or tooling to the project. It complements the [official Contributor's Guide](https://git.k8s.io/community/contributors/guide/) and the resources in the **kubernetes/community** repository.

---

## Table of Contents
1. [Signing the CLA](#signing-the-cla)
2. [Getting the Source Code](#getting-the-source-code)
3. [Setting Up Your Development Environment](#setting-up-your-development-environment)
4. [Building the Repository](#building-the-repository)
5. [Running Tests](#running-tests)
6. [Code Formatting & Linting](#code-formatting--linting)
7. [Submitting a Pull Request](#submitting-a-pull-request)
8. [Review Process & Merge Guidelines](#review-process--merge-guidelines)
9. [Commonly Modified Areas in This Repo](#commonly-modified-areas-in-this-repo)
10. [Helpful Commands & Scripts](#helpful-commands--scripts)
11. [Community & Governance](#community--governance)
12. [Resources & Further Reading](#resources--further-reading)
---

## Signing the CLA

You must sign the [Contributor License Agreement (CLA)](https://git.k8s.io/community/contributors/guide/README.md#sign-the-cla) before your contributions can be merged. The signing process is automated via the **CLA Bot** when you open your first PR. Follow the bot's instructions to complete the signature.

---

## Getting the Source Code

```bash
# Fork the repo on GitHub
# Then clone your fork locally
git clone https://github.com/<YOUR_USERNAME>/kubernetes.git
cd kubernetes
```

The repository is large; consider using the `--depth 1` flag for a shallow clone if you only need the latest tip.

---

## Setting Up Your Development Environment

Kubernetes is written primarily in **Go** (>=1.22) and uses a number of helper tools. The easiest way to get a consistent environment is to use the provided Docker-based dev container, but you can also install the tools locally.

### Prerequisites
- Go 1.22 or newer (`go version` should show the correct version)
- GNU Make
- Docker (for running integration tests that require containers)
- `git`, `curl`, `jq` (standard Unix utilities)

### Recommended: Dev Container
The repository includes a `Dockerfile` and a `devspace.yaml` that can spin up a ready‑to‑code environment:

```bash
# From the repository root
make devcontainer
```

This target builds the `k8sdev` image and starts a container with the source mounted.

---

## Building the Repository

Kubernetes uses a Makefile wrapper around the go toolchain. The most common build targets are:

| Target | Description |
|--------|-------------|
| `make all` | Compiles all binaries (`kube-apiserver`, `kube-controller-manager`, `kubectl`, etc.) |
| `make binaries` | Builds only the release‑ready binaries without tests |
| `make test` | Runs the unit test suite across all packages |
| `make verify` | Runs static analysis, code‑generation checks, and linting |
| `make quick-release` | Builds binaries for the current host and packages them as a local tarball |

> **Tip**: Most contributors run `make test` and `make verify` before pushing changes.

---

## Running Tests

The repo ships with a comprehensive test matrix. Use the following shortcuts:

- **Unit tests** (fast, no external dependencies):
  ```bash
  make test
  ```
- **Integration tests** that require a real cluster or Docker containers. Examples include tests under `cluster/gce/gci/` and `build/pause/windows/`:
  ```bash
  # Run all integration tests (may take >30 min)
  make test-integration
  ```
- **Specific package** – to run tests for a single package, use `go test ./path/to/pkg` or the Makefile helper:
  ```bash
  make test WHAT=./cluster/gce/gci
  ```
- **Race detector** – useful for catching data races:
  ```bash
  make test-race
  ```

All test output is written to the standard Go testing format, which CI tools (Prow, Jenkins) consume directly.

---

## Code Formatting & Linting

Kubernetes enforces a strict style. Before opening a PR, ensure the following checks pass:

1. **Go formatting** – `go fmt ./...` (or `make fmt`).
2. **Static analysis** – `make verify` runs `go vet`, `golint`, `staticcheck`, `shadow`, and other linters.
3. **License headers** – every source file must contain the Apache 2.0 boilerplate. Run `make verify-license` to validate.
4. **Generated code** – if you modify API types or add new flags, run the appropriate generators (e.g., `make gen`). The `make verify` target will flag out‑of‑date generated files.

---

## Submitting a Pull Request

1. **Create a topic branch** (descriptive, prefixed with `feature/` or `bug/`):
   ```bash
   git checkout -b feature/add-wincat-tool
   ```
2. **Make your changes** – keep changes focused to a single logical unit.
3. **Run the full verification suite**:
   ```bash
   make verify
   make test
   ```
4. **Push and open a PR** against the `master` branch of the upstream `kubernetes/kubernetes` repo.
5. **Fill the PR template** – include:
   - A clear title (`docs: Update CONTRIBUTING.md for dev env changes` etc.)
   - A concise description of what and why.
   - Links to related issues (use `Fixes #12345` to auto‑close).
   - Test plan and any required manual steps.
6. **Add reviewers** – you can tag teams via `@kubernetes/sig-api-machinery` or similar, but the automatic reviewer assignment in Prow will also work.

---

## Review Process & Merge Guidelines

- **Reviewers** must sign‑off that the change adheres to style, passes all CI jobs, and includes appropriate tests.
- **Approvals**: at least **two** LGTM approvals from reviewers with write access are required for most changes. Some SIG‑specific changes may have additional owner approvals.
- **Merge method**: the repository uses **squash‑merge** to keep a linear history. The PR description should contain a concise commit message.
- **Breaking changes** must be accompanied by:
  - A deprecation notice in the relevant API documentation.
  - A clear migration guide under `CHANGELOG/` or the appropriate `doc/` directory.
  - An entry in the `kubernetes/kubernetes/` *v1.XX* release notes.

---

## Commonly Modified Areas in This Repo

| Path | Typical Changes |
|------|-----------------|
| `cmd/` | New CLI tools (`cmd/clicheck`, `cmd/dependencycheck`, `cmd/gen*`) – update generated docs via `make gen` or `make gen-docs`. |
| `build/` | Platform‑specific build helpers (e.g., `build/pause/windows/wincat/wincat.go`). Ensure `make verify` picks up any new binaries. |
| `cluster/gce/gci/` | Test suites that interact with GCE; changes often require updating the GCE test harness or adding new fixture data. |
| `pkg/` (not listed but large) | API type changes – run `make generated_files` and update OpenAPI specs. |
| `cmd/gendocs/` | Documentation generation – run `make gen-kubectl-docs` after code changes. |

---

## Helpful Commands & Scripts

```bash
# Quick sanity check – format, lint, and unit tests
make verify && make test

# Regenerate all generated files (clientsets, deepcopy, docs)
make generated_files

# Run the CI locally (requires Docker)
hack/ci/run-ci.sh

# List all owners for a file (useful for finding reviewers)
go run ./cmd/owners/main.go -path=cmd/cloud-controller-manager/main.go
```

---

## Community & Governance

- **SIGs** (Special Interest Groups) own large sections of the code. Find the right SIG for your change in the `OWNERS` files or on the [SIG docs page](https://git.k8s.io/community/sig-list.md).
- **Code of Conduct** – all contributors must follow the Kubernetes [Code of Conduct](https://github.com/kubernetes/community/blob/master/code-of-conduct.md).
- **Issue Tracker** – use GitHub Issues for bug reports and feature requests. Tag the appropriate SIG and use the issue templates provided.

---

## Resources & Further Reading

- **Official Contributor Guide** – https://git.k8s.io/community/contributors/guide/
- **Kubernetes Development Guide** – https://kubernetes.io/docs/contribute/
- **API Review Process** – https://git.k8s.io/community/sig-api-machinery/api-review-process.md
- **Testing Guide** – https://git.k8s.io/community/contributors/guide/testing.md
- **Prow CI** – https://github.com/kubernetes/test-infra/tree/master/prow

Thank you for your interest in contributing to Kubernetes! Your effort helps make the platform stronger for everyone.
