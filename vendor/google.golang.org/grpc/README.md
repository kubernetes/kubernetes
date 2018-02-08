# gRPC-Go

[![Build Status](https://travis-ci.org/grpc/grpc-go.svg)](https://travis-ci.org/grpc/grpc-go) [![GoDoc](https://godoc.org/google.golang.org/grpc?status.svg)](https://godoc.org/google.golang.org/grpc)

The Go implementation of [gRPC](https://grpc.io/): A high performance, open source, general RPC framework that puts mobile and HTTP/2 first. For more information see the [gRPC Quick Start: Go](https://grpc.io/docs/quickstart/go.html) guide.

Installation
------------

To install this package, you need to install Go and setup your Go workspace on your computer. The simplest way to install the library is to run:

```
$ go get -u google.golang.org/grpc
```

Prerequisites
-------------

This requires Go 1.7 or later.

Constraints
-----------
The grpc package should only depend on standard Go packages and a small number of exceptions. If your contribution introduces new dependencies which are NOT in the [list](http://godoc.org/google.golang.org/grpc?imports), you need a discussion with gRPC-Go authors and consultants.

Documentation
-------------
See [API documentation](https://godoc.org/google.golang.org/grpc) for package and API descriptions and find examples in the [examples directory](examples/).

Performance
-----------
See the current benchmarks for some of the languages supported in [this dashboard](https://performance-dot-grpc-testing.appspot.com/explore?dashboard=5652536396611584&widget=490377658&container=1286539696).

Status
------
General Availability [Google Cloud Platform Launch Stages](https://cloud.google.com/terms/launch-stages).

FAQ
---

#### Compiling error, undefined: grpc.SupportPackageIsVersion

Please update proto package, gRPC package and rebuild the proto files:
 - `go get -u github.com/golang/protobuf/{proto,protoc-gen-go}`
 - `go get -u google.golang.org/grpc`
 - `protoc --go_out=plugins=grpc:. *.proto`
