# console

[![Build Status](https://travis-ci.org/containerd/console.svg?branch=master)](https://travis-ci.org/containerd/console)

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
 * [Project governance](https://github.com/containerd/project/blob/master/GOVERNANCE.md),
 * [Maintainers](https://github.com/containerd/project/blob/master/MAINTAINERS),
 * and [Contributing guidelines](https://github.com/containerd/project/blob/master/CONTRIBUTING.md)

information in our [`containerd/project`](https://github.com/containerd/project) repository.
