#gRPC-Go

[![Build Status](https://travis-ci.org/grpc/grpc-go.svg)](https://travis-ci.org/grpc/grpc-go) [![GoDoc](https://godoc.org/google.golang.org/grpc?status.svg)](https://godoc.org/google.golang.org/grpc)

The Go implementation of [gRPC](http://www.grpc.io/): A high performance, open source, general RPC framework that puts mobile and HTTP/2 first. For more information see the [gRPC Quick Start](http://www.grpc.io/docs/) guide.

Installation
------------

To install this package, you need to install Go and setup your Go workspace on your computer. The simplest way to install the library is to run:

```
$ go get google.golang.org/grpc
```

Prerequisites
-------------

This requires Go 1.5 or later.

A note on the version used: significant performance improvements in benchmarks
of grpc-go have been seen by upgrading the go version from 1.5 to the latest
1.7.1.

From https://golang.org/doc/install, one way to install the latest version of go is:
```
$ GO_VERSION=1.7.1
$ OS=linux
$ ARCH=amd64
$ curl -O https://storage.googleapis.com/golang/go${GO_VERSION}.${OS}-${ARCH}.tar.gz
$ sudo tar -C /usr/local -xzf go$GO_VERSION.$OS-$ARCH.tar.gz
$ # Put go on the PATH, keep the usual installation dir
$ sudo ln -s /usr/local/go/bin/go /usr/bin/go
$ rm go$GO_VERSION.$OS-$ARCH.tar.gz
```

Constraints
-----------
The grpc package should only depend on standard Go packages and a small number of exceptions. If your contribution introduces new dependencies which are NOT in the [list](http://godoc.org/google.golang.org/grpc?imports), you need a discussion with gRPC-Go authors and consultants.

Documentation
-------------
See [API documentation](https://godoc.org/google.golang.org/grpc) for package and API descriptions and find examples in the [examples directory](examples/).

Status
------
GA

FAQ
---

#### Compiling error, undefined: grpc.SupportPackageIsVersion

Please update proto package, gRPC package and rebuild the proto files:
 - `go get -u github.com/golang/protobuf/{proto,protoc-gen-go}`
 - `go get -u google.golang.org/grpc`
 - `protoc --go_out=plugins=grpc:. *.proto`
