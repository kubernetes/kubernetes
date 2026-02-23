# Contributing to Kubernetes

Welcome to the **Kubernetes** repository! We appreciate your interest in contributing to one of the most widely used container orchestration platforms. This guide provides the essential steps, best‑practice recommendations, and tooling needed to submit high‑quality contributions.

---

## Table of Contents
1. [Overview](#overview)
2. [Sign the Contributor License Agreement (CLA)](#sign-the-contributor-license-agreement-cla)
3. [Getting Started](#getting-started)
4. [Development Environment Setup](#development-environment-setup)
5. [Building the Code](#building-the-code)
6. [Running Tests](#running-tests)
7. [Submitting a Pull Request](#submitting-a-pull-request)
8. [Review Process & Merging](#review-process--merging)
9. [Code Style & Linting](#code-style--linting)
10. [Additional Resources](#additional-resources)
11. [Code of Conduct](#code-of-conduct)
---

## Overview
Kubernetes is an open‑source project governed by a large, global community. Contributions can take many forms, including:
- New features or enhancements to existing components (e.g., `cmd/cloud-controller-manager`, `cmd/kubectl`, `build/*` utilities)
- Bug fixes and regression tests
- Documentation improvements (README, design docs, API docs, etc.)
- Test suites and continuous‑integration tooling
- Build and release tooling (e.g., `cmd/dependencycheck`, `cmd/gendocs`)

All contributions must follow the processes outlined below to ensure a consistent, high‑quality code base.

## Sign the Contributor License Agreement (CLA)
Before any contribution can be merged, you must sign the **Contributor License Agreement**. The CLA grants the Kubernetes project the rights to safely use and redistribute your contributions.

- Read the CLA details: https://git.k8s.io/community/contributors/guide/README.md#sign-the-cla
- Sign the CLA using the web‑based flow (GitHub OAuth will prompt you when you open a PR for the first time).

> **Note:** If your organization has a corporate CLA, follow the instructions provided in the link above.

---

## Getting Started
1. **Fork the repository**
   - Click the **Fork** button on https://github.com/kubernetes/kubernetes.
2. **Clone your fork**
   ```bash
   git clone https://github.com/<your‑username>/kubernetes.git
   cd kubernetes
   ```
3. **Create a new branch** for your work. Use a descriptive name, e.g., `feat/cloud‑controller‑manager‑node‑ipam`.
   ```bash
   git checkout -b feat/cloud‑controller‑manager‑node‑ipam
   ```

---

## Development Environment Setup
Kubernetes is a large Go codebase (Go 1.22+). The following steps install the required tooling on a typical Linux/macOS workstation.

```bash
# Install Go (version must match the version in the repository's .go-version file)
# Example for Go 1.22:
wget https://golang.org/dl/go1.22.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Verify installation
go version

# Install build dependencies (make, gcc, etc.)
# On Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y build-essential git make

# Install additional Kubernetes tooling (optional but recommended)
GO111MODULE=on go install k8s.io/kube-proxy/tools/hack/...@latest
```

### Repository Configuration
- **Git hooks**: The repository ships a `pre-commit` hook that runs `make verify`. Enable it via:
  ```bash
  ln -s ../../hack/.git/hooks/pre-commit .git/hooks/pre-commit
  ```
- **Modules**: The repo uses Go modules. Avoid `GOPATH`‑based workflows.

---

## Building the Code
Kubernetes uses `make` as the primary build orchestrator.

```bash
# Compile the main binary (e.g., kube-apiserver, kubelet, kubectl)
make all

# Build a specific component, such as the cloud‑controller‑manager
make WHAT=cmd/cloud-controller-manager
```

The `build/` directory contains auxiliary tools (`build/tools.go`, `build/pause/windows/wincat/wincat.go`) that are compiled automatically when needed.

---

## Running Tests
The project has an extensive test suite covering unit, integration, and e2e tests.

### Unit Tests
```bash
# Run all Go unit tests (fast)
make test
```

### Integration Tests
Specific packages include integration tests. For example, the GCE GCI tests can be executed with:
```bash
go test ./cluster/gce/gci -run TestApiserverEtcd
```

### End‑to‑End (e2e) Tests
Follow the e2e testing guide in the Kubernetes documentation for cluster‑wide validation. The `hack/` scripts facilitate running e2e tests against a local KIND cluster.

> **Tip:** Before opening a PR, ensure `make verify` succeeds. This command runs static analysis, code generation checks, and linting.

---

## Submitting a Pull Request
1. Push your branch to your fork:
   ```bash
   git push origin feat/cloud‑controller‑manager‑node‑ipam
   ```
2. Open a Pull Request (PR) on the upstream `kubernetes/kubernetes` repository.
3. Fill out the PR template:
   - Provide a concise title (e.g., `feat: add node‑IPAM controller to cloud‑controller‑manager`).
   - Explain the motivation, implementation details, and any relevant design docs.
   - List the test coverage you added or updated.
4. Add the appropriate **OWNERS** reviewers. The repository uses **OWNERS** files scattered throughout the tree (e.g., `cmd/cloud-controller-manager/OWNERS`). If unsure, tag the `sig-cloud-provider` team.
5. Ensure the CI checks pass (unit tests, `make verify`, `make lint`, `make vet`).

---

## Review Process & Merging
- **Reviewers**: At least two maintainers must approve the PR. Reviews focus on correctness, test coverage, and adherence to style guidelines.
- **Merge Method**: Only **squash‑merge** is allowed for most contributions to keep history linear. The `release‑note` label triggers automatic release‑note generation.
- **Breaking Changes**: If your change introduces a breaking API change, include a `k/k` deprecation notice and add a bump to the `api/` version if applicable.

---

## Code Style & Linting
Kubernetes enforces a strict Go style using the following tools:
- `golangci-lint` (run via `make lint`)
- `go vet` (`make vet`)
- `staticcheck` (`make staticcheck`)
- `go fmt` (automatically checked by `make verify`)

When editing files under `cmd/` (e.g., `cmd/cloud-controller-manager/main.go`) or `build/` utilities, follow the existing formatting and comment conventions. Use `cmd/gendocs` to regenerate Markdown API docs after significant changes.

---

## Additional Resources
- **Contributor Guide** – https://git.k8s.io/community/contributors/guide/
- **SIG Documentation** – https://git.k8s.io/community/sig-documentation/
- **Kubernetes Issue Tracker** – https://github.com/kubernetes/kubernetes/issues
- **Roadmap & Release Process** – https://github.com/kubernetes/community/tree/master/sig-release
- **Testing Guide** – https://git.k8s.io/community/contributors/testing.md
- **Code of Conduct** – https://github.com/kubernetes/kubernetes/blob/master/CONTRIBUTING.md#code-of-conduct

---

## Code of Conduct
All participants must adhere to the Kubernetes Code of Conduct. Harassment or discriminatory behavior is not tolerated. See the full policy in `CODE_OF_CONDUCT.md` for details.

---

*Happy hacking!*
