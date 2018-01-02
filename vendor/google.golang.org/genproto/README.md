Go generated proto packages
===========================

[![Build Status](https://travis-ci.org/google/go-genproto.svg?branch=master)](https://travis-ci.org/google/go-genproto)
[![GoDoc](https://godoc.org/google.golang.org/genproto?status.svg)](https://godoc.org/google.golang.org/genproto)

> **IMPORTANT** This repository is currently experimental. The structure
> of the contained packages is subject to change. Please see the original
> source repositories (listed below) to find out the status of the each
> protocol buffer's associated service.

This repository contains the generated Go packages for common protocol buffer
types, and the generated [gRPC][1] code necessary for interacting with Google's gRPC
APIs.

There are two sources for the proto files used in this repository:

1. [google/protobuf][2]: the code in the `protobuf` and `ptypes` subdirectories
   is derived from this repo. The messages in `protobuf` are used to describe
   protocol buffer messages themselves. The messages under `ptypes` define the
   common well-known types.
2. [googleapis/googleapis][3]: the code in the `googleapis` is derived from this
   repo. The packages here contain types specifically for interacting with Google
   APIs.

[1]: http://grpc.io
[2]: https://github.com/google/protobuf/
[3]: https://github.com/googleapis/googleapis/
