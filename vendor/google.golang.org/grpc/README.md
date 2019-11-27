# gRPC-Go

[![Build Status](https://travis-ci.org/grpc/grpc-go.svg)](https://travis-ci.org/grpc/grpc-go)
[![GoDoc](https://godoc.org/google.golang.org/grpc?status.svg)](https://godoc.org/google.golang.org/grpc)
[![GoReportCard](https://goreportcard.com/badge/grpc/grpc-go)](https://goreportcard.com/report/github.com/grpc/grpc-go)

The Go implementation of [gRPC](https://grpc.io/): A high performance, open
source, general RPC framework that puts mobile and HTTP/2 first. For more
information see the [gRPC Quick Start:
Go](https://grpc.io/docs/quickstart/go.html) guide.

Installation
------------

To install this package, you need to install Go and setup your Go workspace on
your computer. The simplest way to install the library is to run:

```
$ go get -u google.golang.org/grpc
```

With Go module support (Go 1.11+), simply `import "google.golang.org/grpc"` in
your source code and `go [build|run|test]` will automatically download the
necessary dependencies ([Go modules
ref](https://github.com/golang/go/wiki/Modules)).

If you are trying to access grpc-go from within China, please see the
[FAQ](#FAQ) below.

Prerequisites
-------------
gRPC-Go requires Go 1.9 or later.

Documentation
-------------
- See [godoc](https://godoc.org/google.golang.org/grpc) for package and API
  descriptions.
- Documentation on specific topics can be found in the [Documentation
  directory](Documentation/).
- Examples can be found in the [examples directory](examples/).

Performance
-----------
Performance benchmark data for grpc-go and other languages is maintained in
[this
dashboard](https://performance-dot-grpc-testing.appspot.com/explore?dashboard=5652536396611584&widget=490377658&container=1286539696).

Status
------
General Availability [Google Cloud Platform Launch
Stages](https://cloud.google.com/terms/launch-stages).

FAQ
---

#### I/O Timeout Errors

The `golang.org` domain may be blocked from some countries.  `go get` usually
produces an error like the following when this happens:

```
$ go get -u google.golang.org/grpc
package google.golang.org/grpc: unrecognized import path "google.golang.org/grpc" (https fetch: Get https://google.golang.org/grpc?go-get=1: dial tcp 216.239.37.1:443: i/o timeout)
```

To build Go code, there are several options:

- Set up a VPN and access google.golang.org through that.

- Without Go module support: `git clone` the repo manually:

  ```
  git clone https://github.com/grpc/grpc-go.git $GOPATH/src/google.golang.org/grpc
  ```

  You will need to do the same for all of grpc's dependencies in `golang.org`,
  e.g. `golang.org/x/net`.

- With Go module support: it is possible to use the `replace` feature of `go
  mod` to create aliases for golang.org packages.  In your project's directory:

  ```
  go mod edit -replace=google.golang.org/grpc=github.com/grpc/grpc-go@latest
  go mod tidy
  go mod vendor
  go build -mod=vendor
  ```

  Again, this will need to be done for all transitive dependencies hosted on
  golang.org as well.  Please refer to [this
  issue](https://github.com/golang/go/issues/28652) in the golang repo regarding
  this concern.

#### Compiling error, undefined: grpc.SupportPackageIsVersion

Please update proto package, gRPC package and rebuild the proto files:
 - `go get -u github.com/golang/protobuf/{proto,protoc-gen-go}`
 - `go get -u google.golang.org/grpc`
 - `protoc --go_out=plugins=grpc:. *.proto`

#### How to turn on logging

The default logger is controlled by the environment variables. Turn everything
on by setting:

```
GRPC_GO_LOG_VERBOSITY_LEVEL=99 GRPC_GO_LOG_SEVERITY_LEVEL=info
```

#### The RPC failed with error `"code = Unavailable desc = transport is closing"`

This error means the connection the RPC is using was closed, and there are many
possible reasons, including:
 1. mis-configured transport credentials, connection failed on handshaking
 1. bytes disrupted, possibly by a proxy in between
 1. server shutdown

It can be tricky to debug this because the error happens on the client side but
the root cause of the connection being closed is on the server side. Turn on
logging on __both client and server__, and see if there are any transport
errors.
