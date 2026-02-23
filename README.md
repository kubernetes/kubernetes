# Kubernetes (K8s)

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/569/badge)](https://bestpractices.coreinfrastructure.org/projects/569) [![Go Report Card](https://goreportcard.com/badge/github.com/kubernetes/kubernetes)](https://goreportcard.com/report/github.com/kubernetes/kubernetes) ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/kubernetes/kubernetes?sort=semver)

<img src="https://github.com/kubernetes/kubernetes/raw/master/logo/logo.png" width="100" alt="Kubernetes logo"/>

---

## Overview

Kubernetes (pronounced *koo‑ber‑net‑ez*, often shortened to **K8s**) is an open‑source platform for automating deployment, scaling, and operations of application containers across clusters of hosts.  It provides a declarative model for describing containerized workloads and a rich set of APIs for managing those workloads.

Originally inspired by Google’s internal Borg system, Kubernetes incorporates a decade‑plus of production experience and a broad ecosystem of open‑source contributions.  It is a graduated project of the Cloud Native Computing Foundation (CNCF).

---

## Installation

### Prerequisites

- **Go** – the repository follows the `go.mod` file; building from source requires the version specified in the `go.mod` (currently Go 1.22).
- **Docker** – required for building images used by `make` targets such as `make test` and `make quick-release`.
- **GNU Make** – the build system is driven by Makefiles throughout the repo.
- **Git** – to clone the source tree.

### Clone the repository

```bash
git clone https://github.com/kubernetes/kubernetes.git
cd kubernetes
```

### Build the binaries

The repository provides a top‑level `Makefile` with a collection of targets.  The most common workflow is:

```bash
# Build all core components (kube-apiserver, kube-controller‑manager, …)
make all
```

Artifacts are placed in the `_output/bin` directory:

- `kube-apiserver`
- `kube-controller-manager`
- `kube-scheduler`
- `kubelet`
- `kubectl`
- `kube-proxy`
- `cloud-controller-manager`

You can also build a single binary, e.g.:

```bash
make WHAT=cmd/kubelet/kubelet
```

### Quick‑release (optional)

For developers who want a pre‑built release binary without pulling Docker images:

```bash
make quick-release
```

The resulting tarball is written to `_output/dockerized` and contains the same set of binaries.

### Running the components locally

After building, you can start a minimal control‑plane for experimentation:

```bash
# Start etcd (required for the API server)
./_output/bin/etcd &

# Start the API server
./_output/bin/kube-apiserver --etcd-servers=http://127.0.0.1:2379 &

# Start the controller manager and scheduler
./_output/bin/kube-controller-manager &
./_output/bin/kube-scheduler &
```

For a full‑featured development cluster, see the `kind` or `minikube` projects, which consume the binaries built from this repository.

---

## Usage

Kubernetes is primarily interacted with via the `kubectl` command‑line tool.  Once the control‑plane components are up, you can create a simple pod:

```bash
# Create a namespace for testing
kubectl create namespace demo

# Deploy an nginx pod
kubectl run nginx --image=nginx --restart=Never -n demo

# Verify the pod is running
kubectl get pods -n demo
```

The repository also contains a number of command‑line utilities used during development and CI:

- `cmd/cloud-controller-manager/main.go` – the entry point for the cloud‑controller‑manager binary.
- `cmd/dependencycheck` – verifies module dependencies.
- `cmd/gendocs` – generates documentation for `kubectl` and other commands.
- `cmd/genfeaturegates` – produces the feature‑gate documentation file.

Refer to the generated help output (`./_output/bin/kubectl --help`) for the full list of commands and flags.

---

## Contributing

We welcome contributions from the community!  The recommended workflow is:

1. **Read the contributing guide** – the repository contains a comprehensive [CONTRIBUTING.md] that outlines the development process, testing requirements, and code‑style conventions.
2. **Set up a development environment** – follow the *Installation* section above, then run `make test` to ensure your environment matches CI expectations.
3. **Create a fork** of the `kubernetes/kubernetes` repo and push your changes to a new branch.
4. **Open a Pull Request** against the `master` branch.  Include a clear description of the problem being solved, reference any related issues, and add end‑to‑end tests where appropriate.
5. **Engage with reviewers** – CI will run a suite of tests (unit, integration, e2e).  Address feedback promptly; maintainers may request additional documentation or refactoring.

### Community resources

- **Community repository** – https://github.com/kubernetes/community – contains information on governance, communication channels, and how to get involved.
- **Slack** – `#kubernetes-dev` on the CNCF Slack workspace.
- **Mailing lists** – `kubernetes-dev@googlegroups.com` for design discussions.

### Code of Conduct

All participants must adhere to the Kubernetes [Code of Conduct] (see the `CODE_OF_CONDUCT.md` file).

---

## Additional resources

- Official documentation site: https://kubernetes.io/docs/
- Release notes and changelog: https://github.com/kubernetes/kubernetes/releases
- API reference (OpenAPI): https://github.com/kubernetes/kubernetes/tree/master/api/openapi-spec

---

*This README is generated from the source tree at commit `$(git rev-parse HEAD)` and reflects the current state of the repository.*