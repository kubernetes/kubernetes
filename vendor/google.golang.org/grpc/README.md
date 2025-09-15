# gRPC-Go

[![GoDoc](https://pkg.go.dev/badge/google.golang.org/grpc)][API]
[![GoReportCard](https://goreportcard.com/badge/grpc/grpc-go)](https://goreportcard.com/report/github.com/grpc/grpc-go)
[![codecov](https://codecov.io/gh/grpc/grpc-go/graph/badge.svg)](https://codecov.io/gh/grpc/grpc-go)

The [Go][] implementation of [gRPC][]: A high performance, open source, general
RPC framework that puts mobile and HTTP/2 first. For more information see the
[Go gRPC docs][], or jump directly into the [quick start][].

## Prerequisites

- **[Go][]**: any one of the **two latest major** [releases][go-releases].

## Installation

Simply add the following import to your code, and then `go [build|run|test]`
will automatically fetch the necessary dependencies:


```go
import "google.golang.org/grpc"
```

> **Note:** If you are trying to access `grpc-go` from **China**, see the
> [FAQ](#FAQ) below.

## Learn more

- [Go gRPC docs][], which include a [quick start][] and [API
  reference][API] among other resources
- [Low-level technical docs](Documentation) from this repository
- [Performance benchmark][]
- [Examples](examples)

## FAQ

### I/O Timeout Errors

The `golang.org` domain may be blocked from some countries. `go get` usually
produces an error like the following when this happens:

```console
$ go get -u google.golang.org/grpc
package google.golang.org/grpc: unrecognized import path "google.golang.org/grpc" (https fetch: Get https://google.golang.org/grpc?go-get=1: dial tcp 216.239.37.1:443: i/o timeout)
```

To build Go code, there are several options:

- Set up a VPN and access google.golang.org through that.

- With Go module support: it is possible to use the `replace` feature of `go
  mod` to create aliases for golang.org packages.  In your project's directory:

  ```sh
  go mod edit -replace=google.golang.org/grpc=github.com/grpc/grpc-go@latest
  go mod tidy
  go mod vendor
  go build -mod=vendor
  ```

  Again, this will need to be done for all transitive dependencies hosted on
  golang.org as well. For details, refer to [golang/go issue
  #28652](https://github.com/golang/go/issues/28652).

### Compiling error, undefined: grpc.SupportPackageIsVersion

Please update to the latest version of gRPC-Go using
`go get google.golang.org/grpc`.

### How to turn on logging

The default logger is controlled by environment variables. Turn everything on
like this:

```console
$ export GRPC_GO_LOG_VERBOSITY_LEVEL=99
$ export GRPC_GO_LOG_SEVERITY_LEVEL=info
```

### The RPC failed with error `"code = Unavailable desc = transport is closing"`

This error means the connection the RPC is using was closed, and there are many
possible reasons, including:
 1. mis-configured transport credentials, connection failed on handshaking
 1. bytes disrupted, possibly by a proxy in between
 1. server shutdown
 1. Keepalive parameters caused connection shutdown, for example if you have
    configured your server to terminate connections regularly to [trigger DNS
    lookups](https://github.com/grpc/grpc-go/issues/3170#issuecomment-552517779).
    If this is the case, you may want to increase your
    [MaxConnectionAgeGrace](https://pkg.go.dev/google.golang.org/grpc/keepalive?tab=doc#ServerParameters),
    to allow longer RPC calls to finish.

It can be tricky to debug this because the error happens on the client side but
the root cause of the connection being closed is on the server side. Turn on
logging on __both client and server__, and see if there are any transport
errors.

[API]: https://pkg.go.dev/google.golang.org/grpc
[Go]: https://golang.org
[Go module]: https://github.com/golang/go/wiki/Modules
[gRPC]: https://grpc.io
[Go gRPC docs]: https://grpc.io/docs/languages/go
[Performance benchmark]: https://performance-dot-grpc-testing.appspot.com/explore?dashboard=5180705743044608
[quick start]: https://grpc.io/docs/languages/go/quickstart
[go-releases]: https://golang.org/doc/devel/release.html
