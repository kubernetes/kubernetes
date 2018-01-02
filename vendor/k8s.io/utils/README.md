# Utils

[![Build Status]](https://travis-ci.org/kubernetes/utils) [![GoDoc](https://godoc.org/k8s.io/utils?status.svg)](https://godoc.org/k8s.io/utils)

A set of Go libraries that provide low-level,
kubernetes-independent packages supplementing the [Go
standard libs].

## Purpose

As Kubernetes grows and spins functionality out of its
[core] and into cooperating repositories like
[apiserver], [kubectl], [kubeadm], etc., the need
arises for leaf repositories to house shared code and
avoid cycles in repository relationships.

This repository is intended to hold shared utilities
with no Kubernetes dependence that may be of interest
to any Go project.  See these [instructions for moving]
an existing package to this repository.


## Criteria for adding code here

- Used by multiple Kubernetes repositories.

- Full unit test coverage.

- Go tools compliant (`go get`, `go test`, etc.).

- Complex enough to be worth vendoring, rather than copying.

- Stable, or backward compatible, API.

- _No dependence on any Kubernetes repository_.

## Libraries

- [Exec](/exec) provides an interface for `os/exec`. It makes it easier
  to mock and replace in tests, especially with
  the [FakeExec](exec/testing/fake_exec.go) struct.

- [Temp](/temp) provides an interface to create temporary directories. It also
  provides a [FakeDir](temp/temptesting) implementation to replace in tests.

- [Clock](/clock) provides an interface for time-based operations.  It allows
  mocking time for testing.

[Build Status]: https://travis-ci.org/kubernetes/utils.svg?branch=master
[Go standard libs]: https://golang.org/pkg/#stdlib
[api]: https://github.com/kubernetes/api
[apiserver]: https://github.com/kubernetes/apiserver
[core]: https://github.com/kubernetes/kubernetes
[ingress]: https://github.com/kubernetes/ingress
[kubeadm]: https://github.com/kubernetes/kubeadm
[kubectl]: https://github.com/kubernetes/kubectl
[instructions for moving]: ./HOWTOMOVE.md
