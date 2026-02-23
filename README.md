# Kubernetes (K8s)

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/569/badge)](https://bestpractices.coreinfrastructure.org/projects/569) [![Go Report Card](https://goreportcard.com/badge/github.com/kubernetes/kubernetes)](https://goreportcard.com/report/github.com/kubernetes/kubernetes) ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/kubernetes/kubernetes?sort=semver)

<img src="https://github.com/kubernetes/kubernetes/raw/master/logo/logo.png" width="100" alt="Kubernetes logo" />

---

## Overview
Kubernetes (sometimes abbreviated **K8s**) is an open‑source platform for automating deployment, scaling, and operation of containerised applications.  It abstracts away the underlying infrastructure—bare‑metal servers, virtual machines, or public‑cloud instances—so developers can focus on building resilient services.

The project originated from Google’s internal **Borg** system and has since been re‑implemented by the community under the umbrella of the **Cloud Native Computing Foundation (CNCF)**.  Kubernetes provides a rich set of APIs, a declarative model, and a vibrant ecosystem of extensions.

---

## Quick Start (Installation)

### Prerequisites
- **Go** (the version required to build the current master is listed in `go.mod`; as of this writing it is Go 1.22 or later).
- **Docker** (or another container runtime) for building images used by the test harness.
- **Make** – the repository ships a `Makefile` that drives the build, test, and release processes.

### Build from source
```sh
# Clone the repository
git clone https://github.com/kubernetes/kubernetes.git
cd kubernetes

# Compile all core binaries (kube‑apiserver, kube‑controller‑manager, kube‑scheduler, kubelet, kubectl, etc.)
make
```
The `make` target builds the following executables and places them in `_output/bin/`:
- `kube-apiserver`
- `kube-controller-manager`
- `kube-scheduler`
- `kubelet`
- `kubectl`
- `cloud-controller-manager`
- several helper tools under `cmd/` (e.g., `dependencycheck`, `gendocs`, `genfeaturegates`).

### Release binaries (optional)
Pre‑built binaries for each release are published on the [GitHub releases page](https://github.com/kubernetes/kubernetes/releases).  Download the appropriate archive for your platform and extract the binaries to a directory on your `$PATH`.

---

## Usage

### Running a local cluster
Kubernetes ships a lightweight script for spinning up a single‑node cluster useful for development and testing:
```sh
# From the repository root
./hack/local-up-cluster.sh
```
The script starts `kube-apiserver`, `etcd`, `kube-controller-manager`, `kube-scheduler`, and a `kubelet` that runs workloads locally.

For more ergonomic local clusters, consider using tools built on top of this repository such as **kind** or **minikube** – they pull the binaries produced by this repo.

### Core command‑line tools
- **`kubectl`** – the primary CLI for interacting with a Kubernetes cluster. Example:
  ```sh
  ./_output/bin/kubectl get nodes
  ```
- **`kube-apiserver`**, **`kube-controller-manager`**, **`kube-scheduler`**, **`kubelet`** – the control‑plane and node components.  These are typically started by systemd units in a production deployment.
- **`cloud-controller-manager`** – runs cloud‑provider specific control loops. See `cmd/cloud-controller-manager/main.go` for the entry point.
- **Utility commands** located under `cmd/`:
  - `dependencycheck` – validates module and binary dependencies.
  - `gendocs` – generates markdown documentation for `kubectl` and other components.
  - `genfeaturegates` – produces the feature‑gate registry.
  - `fieldnamedocscheck` – ensures API field documentation is present.
  - `clicheck` – enforces CLI conventions across the code base.

---

## Development

### Repository layout (high‑level)
- **`cmd/`** – entry points for the various binaries listed above.
- **`pkg/`** – reusable libraries that implement core Kubernetes functionality.
- **`staging/`** – sub‑repositories that are versioned independently but vendored into the main repo (e.g., client‑go, apimachinery).
- **`test/`** – integration and end‑to‑end test suites.
- **`cluster/`** – scripts and configuration for provisioning clusters on cloud providers (e.g., GCE, GKE).
- **`hack/`** – developer utilities, including the `local-up-cluster.sh` script.

### Building individual binaries
You can build a single component without compiling the whole tree, for example:
```sh
make WHAT=cmd/kubelet
```
The binary will appear at `_output/bin/kubelet`.

### Running tests
```sh
# Run the full test matrix (may take >30 minutes)
make test

# Run a specific test package, e.g., GCE GCI tests
go test ./cluster/gce/gci/... -v
```
The repository also includes linting and static‑analysis tools that are invoked via `make sanity`.

### Contributing code
1. Fork the repository and create a feature branch.
2. Follow the style guidelines in `hack/` (run `make verify` locally).
3. Write unit tests covering new or changed functionality.
4. Submit a Pull Request against the `master` branch.
5. Ensure CI passes and address reviewer feedback.

For detailed contribution guidelines, see the **CONTRIBUTING.md** file and the [Kubernetes community repository](https://github.com/kubernetes/community).

---

## Contributing & Community
Kubernetes is a community‑driven project governed by the CNCF.  Everyone is welcome to contribute:
- **Pull Requests** – see `CONTRIBUTING.md` for the full workflow.
- **Design Proposals** – large‑scale changes should be discussed as a KEP (Kubernetes Enhancement Proposal).
- **Code of Conduct** – we enforce a respectful environment; read the CODE_OF_CONDUCT.md file.
- **Community resources** – the [Kubernetes community repo](https://github.com/kubernetes/community) contains documentation on SIGs, mailing lists, meetings, and the roadmap.

---

## License
Kubernetes is licensed under the Apache License, Version 2.0.  See the `LICENSE` file for the full text.
