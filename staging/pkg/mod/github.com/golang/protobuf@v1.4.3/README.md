# Go support for Protocol Buffers

[![GoDev](https://img.shields.io/static/v1?label=godev&message=reference&color=00add8)](https://pkg.go.dev/mod/github.com/golang/protobuf)
[![Build Status](https://travis-ci.org/golang/protobuf.svg?branch=master)](https://travis-ci.org/golang/protobuf)

This module
([`github.com/golang/protobuf`](https://pkg.go.dev/mod/github.com/golang/protobuf))
contains Go bindings for protocol buffers.

It has been superseded by the
[`google.golang.org/protobuf`](https://pkg.go.dev/mod/google.golang.org/protobuf)
module, which contains an updated and simplified API,
support for protobuf reflection, and many other improvements.
We recommend that new code use the `google.golang.org/protobuf` module.

Versions v1.4 and later of `github.com/golang/protobuf` are implemented
in terms of `google.golang.org/protobuf`.
Programs which use both modules must use at least version v1.4 of this one.

See the
[developer guide for protocol buffers in Go](https://developers.google.com/protocol-buffers/docs/gotutorial)
for a general guide for how to get started using protobufs in Go.

See
[release note documentation](https://github.com/golang/protobuf/releases)
for more information about individual releases of this project.

See
[documentation for the next major revision](https://pkg.go.dev/mod/google.golang.org/protobuf)
for more information about the purpose, usage, and history of this project.

## Package index

Summary of the packages provided by this module:

*   [`proto`](https://pkg.go.dev/github.com/golang/protobuf/proto): Package
    `proto` provides functions operating on protobuf messages such as cloning,
    merging, and checking equality, as well as binary serialization and text
    serialization.
*   [`jsonpb`](https://pkg.go.dev/github.com/golang/protobuf/jsonpb): Package
    `jsonpb` serializes protobuf messages as JSON.
*   [`ptypes`](https://pkg.go.dev/github.com/golang/protobuf/ptypes): Package
    `ptypes` provides helper functionality for protobuf well-known types.
*   [`ptypes/any`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/any):
    Package `any` is the generated package for `google/protobuf/any.proto`.
*   [`ptypes/empty`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/empty):
    Package `empty` is the generated package for `google/protobuf/empty.proto`.
*   [`ptypes/timestamp`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/timestamp):
    Package `timestamp` is the generated package for
    `google/protobuf/timestamp.proto`.
*   [`ptypes/duration`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/duration):
    Package `duration` is the generated package for
    `google/protobuf/duration.proto`.
*   [`ptypes/wrappers`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/wrappers):
    Package `wrappers` is the generated package for
    `google/protobuf/wrappers.proto`.
*   [`ptypes/struct`](https://pkg.go.dev/github.com/golang/protobuf/ptypes/struct):
    Package `structpb` is the generated package for
    `google/protobuf/struct.proto`.
*   [`protoc-gen-go/descriptor`](https://pkg.go.dev/github.com/golang/protobuf/protoc-gen-go/descriptor):
    Package `descriptor` is the generated package for
    `google/protobuf/descriptor.proto`.
*   [`protoc-gen-go/plugin`](https://pkg.go.dev/github.com/golang/protobuf/protoc-gen-go/plugin):
    Package `plugin` is the generated package for
    `google/protobuf/compiler/plugin.proto`.
*   [`protoc-gen-go`](https://pkg.go.dev/github.com/golang/protobuf/protoc-gen-go):
    The `protoc-gen-go` binary is a protoc plugin to generate a Go protocol
    buffer package.

## Reporting issues

The issue tracker for this project
[is located here](https://github.com/golang/protobuf/issues).

Please report any issues with a sufficient description of the bug or feature
request. Bug reports should ideally be accompanied by a minimal reproduction of
the issue. Irreproducible bugs are difficult to diagnose and fix (and likely to
be closed after some period of time). Bug reports must specify the version of
the
[Go protocol buffer module](https://github.com/protocolbuffers/protobuf-go/releases)
and also the version of the
[protocol buffer toolchain](https://github.com/protocolbuffers/protobuf/releases)
being used.

## Contributing

This project is open-source and accepts contributions. See the
[contribution guide](https://github.com/golang/protobuf/blob/master/CONTRIBUTING.md)
for more information.

## Compatibility

This module and the generated code are expected to be stable over time. However,
we reserve the right to make breaking changes without notice for the following
reasons:

*   **Security:** A security issue in the specification or implementation may
    come to light whose resolution requires breaking compatibility. We reserve
    the right to address such issues.
*   **Unspecified behavior:** There are some aspects of the protocol buffer
    specification that are undefined. Programs that depend on unspecified
    behavior may break in future releases.
*   **Specification changes:** It may become necessary to address an
    inconsistency, incompleteness, or change in the protocol buffer
    specification, which may affect the behavior of existing programs. We
    reserve the right to address such changes.
*   **Bugs:** If a package has a bug that violates correctness, a program
    depending on the buggy behavior may break if the bug is fixed. We reserve
    the right to fix such bugs.
*   **Generated additions**: We reserve the right to add new declarations to
    generated Go packages of `.proto` files. This includes declared constants,
    variables, functions, types, fields in structs, and methods on types. This
    may break attempts at injecting additional code on top of what is generated
    by `protoc-gen-go`. Such practice is not supported by this project.
*   **Internal changes**: We reserve the right to add, modify, and remove
    internal code, which includes all unexported declarations, the
    [`generator`](https://pkg.go.dev/github.com/golang/protobuf/protoc-gen-go/generator)
    package, and all packages under
    [`internal`](https://pkg.go.dev/github.com/golang/protobuf/internal).

Any breaking changes outside of these will be announced 6 months in advance to
[protobuf@googlegroups.com](https://groups.google.com/forum/#!forum/protobuf).
