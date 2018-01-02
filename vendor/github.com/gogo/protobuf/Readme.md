# Protocol Buffers for Go with Gadgets

[![Build Status](https://travis-ci.org/gogo/protobuf.svg?branch=master)](https://travis-ci.org/gogo/protobuf)

gogoprotobuf is a fork of <a href="https://github.com/golang/protobuf">golang/protobuf</a> with extra code generation features.

This code generation is used to achieve:

  - fast marshalling and unmarshalling
  - more canonical Go structures
  - goprotobuf compatibility
  - less typing by optionally generating extra helper code
  - peace of mind by optionally generating test and benchmark code
  - other serialization formats

Keeping track of how up to date gogoprotobuf is relative to golang/protobuf is done in this
<a href="https://github.com/gogo/protobuf/issues/191">issue</a>

## Users

These projects use gogoprotobuf:

  - <a href="http://godoc.org/github.com/coreos/etcd">etcd</a> - <a href="https://blog.gopheracademy.com/advent-2015/etcd-distributed-key-value-store-with-grpc-http2/">blog</a> - <a href="https://github.com/coreos/etcd/blob/master/etcdserver/etcdserverpb/etcdserver.proto">sample proto file</a>
  - <a href="https://www.spacemonkey.com/">spacemonkey</a> - <a href="https://www.spacemonkey.com/blog/posts/go-space-monkey">blog</a>
  - <a href="http://badoo.com">badoo</a> - <a href="https://github.com/badoo/lsd/blob/32061f501c5eca9c76c596d790b450501ba27b2f/proto/lsd.proto">sample proto file</a>
  - <a href="https://github.com/mesos/mesos-go">mesos-go</a> - <a href="https://github.com/mesos/mesos-go/blob/f9e5fb7c2f50ab5f23299f26b6b07c5d6afdd252/api/v0/mesosproto/authentication.proto">sample proto file</a>
  - <a href="https://github.com/mozilla-services/heka">heka</a> - <a href="https://github.com/mozilla-services/heka/commit/eb72fbf7d2d28249fbaf8d8dc6607f4eb6f03351">the switch from golang/protobuf to gogo/protobuf when it was still on code.google.com</a>
  - <a href="https://github.com/cockroachdb/cockroach">cockroachdb</a> - <a href="https://github.com/cockroachdb/cockroach/blob/651d54d393e391a30154e9117ab4b18d9ee6d845/roachpb/metadata.proto">sample proto file</a>
  - <a href="https://github.com/jbenet/go-ipfs">go-ipfs</a> - <a href="https://github.com/ipfs/go-ipfs/blob/2b6da0c024f28abeb16947fb452787196a6b56a2/merkledag/pb/merkledag.proto">sample proto file</a>
  - <a href="https://github.com/philhofer/rkive">rkive-go</a> - <a href="https://github.com/philhofer/rkive/blob/e5dd884d3ea07b341321073882ae28aa16dd11be/rpbc/riak_dt.proto">sample proto file</a>
  - <a href="https://www.dropbox.com">dropbox</a>
  - <a href="https://srclib.org/">srclib</a> - <a href="https://github.com/sourcegraph/srclib/blob/6538858f0c410cac5c63440317b8d009e889d3fb/graph/def.proto">sample proto file</a>
  - <a href="http://www.adyoulike.com/">adyoulike</a>
  - <a href="http://www.cloudfoundry.org/">cloudfoundry</a> - <a href="https://github.com/cloudfoundry/bbs/blob/d673710b8c4211037805129944ee4c5373d6588a/models/events.proto">sample proto file</a>
  - <a href="http://kubernetes.io/">kubernetes</a> - <a href="https://github.com/kubernetes/kubernetes/tree/88d8628137f94ee816aaa6606ae8cd045dee0bff/cmd/libs/go2idl">go2idl built on top of gogoprotobuf</a>
  - <a href="https://dgraph.io/">dgraph</a> - <a href="https://github.com/dgraph-io/dgraph/releases/tag/v0.4.3">release notes</a> - <a href="https://discuss.dgraph.io/t/gogoprotobuf-is-extremely-fast/639">benchmarks</a></a>
  - <a href="https://github.com/centrifugal/centrifugo">centrifugo</a> - <a href="https://forum.golangbridge.org/t/centrifugo-real-time-messaging-websocket-or-sockjs-server-v1-5-0-released/2861">release notes</a> - <a href="https://medium.com/@fzambia/centrifugo-protobuf-inside-json-outside-21d39bdabd68#.o3icmgjqd">blog</a>
  - <a href="https://github.com/docker/swarmkit">docker swarmkit</a> - <a href="https://github.com/docker/swarmkit/blob/63600e01af3b8da2a0ed1c9fa6e1ae4299d75edb/api/objects.proto">sample proto file</a>
  - <a href="https://nats.io/">nats.io</a> - <a href="https://github.com/nats-io/go-nats-streaming/blob/master/pb/protocol.proto">go-nats-streaming</a>
  - <a href="https://github.com/pingcap/tidb">tidb</a> - Communication between <a href="https://github.com/pingcap/tipb/blob/master/generate-go.sh#L4">tidb</a> and <a href="https://github.com/pingcap/kvproto/blob/master/generate_go.sh#L3">tikv</a>
  - <a href="https://github.com/AsynkronIT/protoactor-go">protoactor-go</a> - <a href="https://github.com/AsynkronIT/protoactor-go/blob/dev/protobuf/protoc-gen-protoactor/main.go">vanity command</a> that also generates actors from service definitions

Please lets us know if you are using gogoprotobuf by posting on our <a href="https://groups.google.com/forum/#!topic/gogoprotobuf/Brw76BxmFpQ">GoogleGroup</a>.

### Mentioned

  - <a href="http://www.slideshare.net/albertstrasheim/serialization-in-go">Cloudflare - go serialization talk - Albert Strasheim</a>
  - <a href="http://gophercon.sourcegraph.com/post/83747547505/writing-a-high-performance-database-in-go">gophercon</a>
  - <a href="https://github.com/alecthomas/go_serialization_benchmarks">alecthomas' go serialization benchmarks</a>

## Getting Started

There are several ways to use gogoprotobuf, but for all you need to install go and protoc.
After that you can choose:

  - Speed
  - More Speed and more generated code
  - Most Speed and most customization

### Installation

To install it, you must first have Go (at least version 1.6.3) installed (see [http://golang.org/doc/install](http://golang.org/doc/install)).  Go 1.7.1 and 1.8 are continuously tested.

Next, install the standard protocol buffer implementation from [https://github.com/google/protobuf](https://github.com/google/protobuf).
Most versions from 2.3.1 should not give any problems, but 2.6.1, 3.0.2 and 3.2.0 are continuously tested.

### Speed

Install the protoc-gen-gofast binary

    go get github.com/gogo/protobuf/protoc-gen-gofast

Use it to generate faster marshaling and unmarshaling go code for your protocol buffers.

    protoc --gofast_out=. myproto.proto

This does not allow you to use any of the other gogoprotobuf [extensions](https://github.com/gogo/protobuf/blob/master/extensions.md).

### More Speed and more generated code

Fields without pointers cause less time in the garbage collector.
More code generation results in more convenient methods.

Other binaries are also included:

    protoc-gen-gogofast (same as gofast, but imports gogoprotobuf)
    protoc-gen-gogofaster (same as gogofast, without XXX_unrecognized, less pointer fields)
    protoc-gen-gogoslick (same as gogofaster, but with generated string, gostring and equal methods)

Installing any of these binaries is easy.  Simply run:

    go get github.com/gogo/protobuf/proto
    go get github.com/gogo/protobuf/{binary}
    go get github.com/gogo/protobuf/gogoproto

These binaries allow you to using gogoprotobuf [extensions](https://github.com/gogo/protobuf/blob/master/extensions.md).

### Most Speed and most customization

Customizing the fields of the messages to be the fields that you actually want to use removes the need to copy between the structs you use and structs you use to serialize.
gogoprotobuf also offers more serialization formats and generation of tests and even more methods.

Please visit the [extensions](https://github.com/gogo/protobuf/blob/master/extensions.md) page for more documentation.

Install protoc-gen-gogo:

    go get github.com/gogo/protobuf/proto
    go get github.com/gogo/protobuf/jsonpb
    go get github.com/gogo/protobuf/protoc-gen-gogo
    go get github.com/gogo/protobuf/gogoproto

## GRPC

It works the same as golang/protobuf, simply specify the plugin.
Here is an example using gofast:

    protoc --gofast_out=plugins=grpc:. my.proto
