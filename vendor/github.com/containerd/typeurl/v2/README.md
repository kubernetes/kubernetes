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

## Gogo Protobuf Support Deprecation

Support for gogoprotobuf was removed in v2.3.0. The upstream package has been deprecated since 2022 and users of
typeurl should not rely on Gogo Protobuf support anymore. Users which are still transitioning away from it may
continue to use the v2.2 release until that transition is complete. Since v2.2.1, gogo proto support can be
explicitly removed using the `!no_gogo` build tag.
