# Utils

[![Build Status]](https://travis-ci.org/kubernetes/utils) [![GoDoc](https://godoc.org/k8s.io/utils?status.svg)](https://godoc.org/k8s.io/utils)

A set of Go libraries that provide low-level, kubernetes-independent packages
supplementing the [Go standard libs].

## Purpose

As Kubernetes grows and spins functionality out of its [core] and into
cooperating repositories like [apiserver], [kubectl], [kubeadm], etc., the need
arises for leaf repositories to house shared code and avoid cycles in repository
relationships.

This repository is intended to hold shared utilities with _no Kubernetes
dependencies_ that may be of interest to any Go project.  See these [instructions
for moving] an existing package to this repository.

## Criteria for adding code here

- Used by multiple Kubernetes repositories.

- Complex enough to be worth vendoring, rather than copying (e.g. not 5 LOC).

- Can be fully exercised by unit tests (e.g. no dependencies on kernels).

- Has full unit test coverage.

- Stable, or backward compatible, API, with complete godocs.

- Go tools compliant (`go get`, `go test`, etc.).

- Very few (ideally zero) external dependencies.

- _No dependencies on any other Kubernetes repository_.

[Build Status]: https://travis-ci.org/kubernetes/utils.svg?branch=master
[Go standard libs]: https://golang.org/pkg/#stdlib
[api]: https://github.com/kubernetes/api
[apiserver]: https://github.com/kubernetes/apiserver
[core]: https://github.com/kubernetes/kubernetes
[ingress]: https://github.com/kubernetes/ingress
[kubeadm]: https://github.com/kubernetes/kubeadm
[kubectl]: https://github.com/kubernetes/kubectl
[instructions for moving]: ./HOWTOMOVE.md

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.
