# Kubernetes (K8s)

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/569/badge)](https://bestpractices.coreinfrastructure.org/projects/569)
[![Go Report Card](https://goreportcard.com/badge/github.com/kubernetes/kubernetes)](https://goreportcard.com/report/github.com/kubernetes/kubernetes)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/kubernetes/kubernetes?sort=semver)

<img src="https://github.com/kubernetes/kubernetes/raw/master/logo/logo.png" width="100">

----

Kubernetes, also known as K8s, is an open source system for managing [containerized applications]
across multiple hosts. It provides basic mechanisms for the deployment, maintenance,
and scaling of applications.

Kubernetes builds upon a decade and a half of experience at Google running
production workloads at scale using a system called [Borg],
combined with best-of-breed ideas and practices from the community.

Kubernetes is hosted by the Cloud Native Computing Foundation ([CNCF]).
If your company wants to help shape the evolution of
technologies that are container-packaged, dynamically scheduled,
and microservices-oriented, consider joining the CNCF.
For details about who's involved and how Kubernetes plays a role,
read the CNCF [announcement].

----

## To start using K8s

- Visit the official documentation at [kubernetes.io].
- Take a free course: [Scalable Microservices with Kubernetes].
- Note: Use of the `k8s.io/kubernetes` module or its sub-packages as libraries is **not supported**.
  Instead, refer to the [list of published components](https://git.k8s.io/kubernetes/staging/README.md).

## To start developing K8s

The [community repository] contains information about contributing, building Kubernetes, and project organization.

To build Kubernetes:

#### Option 1: With a [Go environment]

```bash
git clone https://github.com/kubernetes/kubernetes
cd kubernetes
make
