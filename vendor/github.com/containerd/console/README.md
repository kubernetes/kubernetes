# console

[![PkgGoDev](https://pkg.go.dev/badge/github.com/containerd/console)](https://pkg.go.dev/github.com/containerd/console)
[![Build Status](https://github.com/containerd/console/workflows/CI/badge.svg)](https://github.com/containerd/console/actions?query=workflow%3ACI)
[![Go Report Card](https://goreportcard.com/badge/github.com/containerd/console)](https://goreportcard.com/report/github.com/containerd/console)

Golang package for dealing with consoles.  Light on deps and a simple API.

## Modifying the current process

```go
current := console.Current()
defer current.Reset()

if err := current.SetRaw(); err != nil {
}
ws, err := current.Size()
current.Resize(ws)
```

## Project details

console is a containerd sub-project, licensed under the [Apache 2.0 license](./LICENSE).
As a containerd sub-project, you will find the:
 * [Project governance](https://github.com/containerd/project/blob/main/GOVERNANCE.md),
 * [Maintainers](https://github.com/containerd/project/blob/main/MAINTAINERS),
 * and [Contributing guidelines](https://github.com/containerd/project/blob/main/CONTRIBUTING.md)

information in our [`containerd/project`](https://github.com/containerd/project) repository.
