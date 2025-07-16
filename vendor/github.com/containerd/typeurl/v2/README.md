# typeurl

[![PkgGoDev](https://pkg.go.dev/badge/github.com/containerd/typeurl)](https://pkg.go.dev/github.com/containerd/typeurl)
[![Build Status](https://github.com/containerd/typeurl/workflows/CI/badge.svg)](https://github.com/containerd/typeurl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/containerd/typeurl/branch/main/graph/badge.svg)](https://codecov.io/gh/containerd/typeurl)
[![Go Report Card](https://goreportcard.com/badge/github.com/containerd/typeurl)](https://goreportcard.com/report/github.com/containerd/typeurl)

A Go package for managing the registration, marshaling, and unmarshaling of encoded types.

This package helps when types are sent over a ttrpc/GRPC API and marshaled as a protobuf [Any](https://pkg.go.dev/google.golang.org/protobuf@v1.27.1/types/known/anypb#Any)

## Project details

**typeurl** is a containerd sub-project, licensed under the [Apache 2.0 license](./LICENSE).
As a containerd sub-project, you will find the:
 * [Project governance](https://github.com/containerd/project/blob/main/GOVERNANCE.md),
 * [Maintainers](https://github.com/containerd/project/blob/main/MAINTAINERS),
 * and [Contributing guidelines](https://github.com/containerd/project/blob/main/CONTRIBUTING.md)

information in our [`containerd/project`](https://github.com/containerd/project) repository.

## Optional

By default, support for gogoproto is available along side the standard Google
protobuf types.
You can choose to leave gogo support out by using the `!no_gogo` build tag.
