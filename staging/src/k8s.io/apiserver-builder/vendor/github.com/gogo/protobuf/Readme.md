# Protocol Buffers for Go with Gadgets

Travis CI Matrix Builds: [![Build Status](https://travis-ci.org/gogo/protobuf.svg?branch=master)](https://travis-ci.org/gogo/protobuf)

### Getting Started (Give me the speed I don't care about the rest)

Install the protoc-gen-gofast binary

    go get github.com/gogo/protobuf/protoc-gen-gofast

Use it to generate faster marshaling and unmarshaling go code for your protocol buffers.

    protoc --gofast_out=. myproto.proto

### Getting started (I have heard about fields without pointers and more code generation)

Other binaries are also included:

    protoc-gen-gogofast (same as gofast, but imports gogoprotobuf)
    protoc-gen-gogofaster (same as gogofast, without XXX_unrecognized, less pointer fields)
    protoc-gen-gogoslick (same as gogofaster, but with generated string, gostring and equal methods)

### Getting started (I want more customization power over fields, speed, other serialization formats and tests, etc.)

Please visit the [homepage](http://gogo.github.io) for more documentation.

### Installation

To install it, you must first have Go (at least version 1.3.3) installed (see [http://golang.org/doc/install](http://golang.org/doc/install)).  Go 1.3.3, 1.4.2, 1.5.3 and 1.6 are continiuosly tested.

Next, install the standard protocol buffer implementation from [https://github.com/google/protobuf](https://github.com/google/protobuf).
Most versions from 2.3.1 should not give any problems, but 2.5.0, 2.6.1 and 3 alpha are continuously tested.

Finally run:

    go get github.com/gogo/protobuf/proto
    go get github.com/gogo/protobuf/protoc-gen-gogo
    go get github.com/gogo/protobuf/gogoproto

### Proto3

Proto3 is supported, but most of the new native types are not supported yet.
[See Proto3 Issue](https://github.com/gogo/protobuf/issues/57) for more details.

### GRPC

It works the same as golang/protobuf, simply specify the plugin.
Here is an example using gofast:

    protoc --gofast_out=plugins=grpc:. my.proto
