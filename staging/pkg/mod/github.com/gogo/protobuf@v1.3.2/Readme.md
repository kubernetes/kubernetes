[GoGo Protobuf looking for new ownership](https://github.com/gogo/protobuf/issues/691)

# Protocol Buffers for Go with Gadgets

[![Build Status](https://github.com/gogo/protobuf/workflows/Continuous%20Integration/badge.svg)](https://github.com/gogo/protobuf/actions)
[![GoDoc](https://godoc.org/github.com/gogo/protobuf?status.svg)](http://godoc.org/github.com/gogo/protobuf)

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

## Release v1.3.0

The project has updated to release v1.3.0. Check out the release notes <a href="https://github.com/gogo/protobuf/releases/tag/v1.3.0">here</a>.

With this new release comes a new internal library version. This means any newly generated *pb.go files generated with the v1.3.0 library will not be compatible with the old library version (v1.2.1). However, current *pb.go files (generated with v1.2.1) should still work with the new library.

Please make sure you manage your dependencies correctly when upgrading your project. If you are still using v1.2.1 and you update your dependencies, one of which could include a new *pb.go (generated with v1.3.0), you could get a compile time error.

Our upstream repo, golang/protobuf, also had to go through this process in order to update their library version.
Here is a link explaining <a href="https://github.com/golang/protobuf/issues/763#issuecomment-442434870">hermetic builds</a>.


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
  - <a href="https://github.com/AsynkronIT/protoactor-go">protoactor-go</a> - <a href="https://github.com/AsynkronIT/protoactor-go/blob/master/protobuf/protoc-gen-protoactor/main.go">vanity command</a> that also generates actors from service definitions
  - <a href="https://containerd.io/">containerd</a> - <a href="https://github.com/containerd/containerd/tree/master/cmd/protoc-gen-gogoctrd">vanity command with custom field names</a> that conforms to the golang convention.
  - <a href="https://github.com/heroiclabs/nakama">nakama</a>
  - <a href="https://github.com/src-d/proteus">proteus</a>
  - <a href="https://github.com/go-graphite">carbonzipper stack</a>
  - <a href="https://sendgrid.com/">sendgrid</a>
  - <a href="https://github.com/zero-os/0-stor">zero-os/0-stor</a>
  - <a href="https://github.com/spacemeshos/go-spacemesh">go-spacemesh</a>
  - <a href="https://github.com/weaveworks/cortex">cortex</a> - <a href="https://github.com/weaveworks/cortex/blob/fee02a59729d3771ef888f7bf0fd050e1197c56e/pkg/ingester/client/cortex.proto">sample proto file</a>
  - <a href="http://skywalking.apache.org/">Apache SkyWalking APM</a> - Istio telemetry receiver based on Mixer bypass protocol
  - <a href="https://github.com/hyperledger/burrow">Hyperledger Burrow</a> - a permissioned DLT framework
  - <a href="https://github.com/iov-one/weave">IOV Weave</a> - a blockchain framework - <a href="https://github.com/iov-one/weave/tree/23f9856f1e316f93cb3d45d92c4c6a0c4810f6bf/spec/gogo">sample proto files</a>

Please let us know if you are using gogoprotobuf by posting on our <a href="https://groups.google.com/forum/#!topic/gogoprotobuf/Brw76BxmFpQ">GoogleGroup</a>.

### Mentioned

  - <a href="http://www.slideshare.net/albertstrasheim/serialization-in-go">Cloudflare - go serialization talk - Albert Strasheim</a>
  - <a href="https://youtu.be/4xB46Xl9O9Q?t=557">GopherCon 2014 Writing High Performance Databases in Go by Ben Johnson</a>
  - <a href="https://github.com/alecthomas/go_serialization_benchmarks">alecthomas' go serialization benchmarks</a>
  - <a href="http://agniva.me/go/2017/11/18/gogoproto.html">Go faster with gogoproto - Agniva De Sarker</a>
  - <a href="https://www.youtube.com/watch?v=CY9T020HLP8">Evolution of protobuf (Gource Visualization) - Landon Wilkins</a>
  - <a href="https://fosdem.org/2018/schedule/event/gopherjs/">Creating GopherJS Apps with gRPC-Web - Johan Brandhorst</a>
  - <a href="https://jbrandhorst.com/post/gogoproto/">So you want to use GoGo Protobuf - Johan Brandhorst</a>
  - <a href="https://jbrandhorst.com/post/grpc-errors/">Advanced gRPC Error Usage - Johan Brandhorst</a>
  - <a href="https://www.udemy.com/grpc-golang/?couponCode=GITHUB10">gRPC Golang Course on Udemy - Stephane Maarek</a>

## Getting Started

There are several ways to use gogoprotobuf, but for all you need to install go and protoc.
After that you can choose:

  - Speed
  - More Speed and more generated code
  - Most Speed and most customization

### Installation

To install it, you must first have Go (at least version 1.6.3 or 1.9 if you are using gRPC) installed (see [http://golang.org/doc/install](http://golang.org/doc/install)).
Latest patch versions of 1.12 and 1.15 are continuously tested.

Next, install the standard protocol buffer implementation from [https://github.com/google/protobuf](https://github.com/google/protobuf).
Most versions from 2.3.1 should not give any problems, but 2.6.1, 3.0.2 and 3.14.0 are continuously tested.

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

These binaries allow you to use gogoprotobuf [extensions](https://github.com/gogo/protobuf/blob/master/extensions.md). You can also use your own binary.

To generate the code, you also need to set the include path properly.

    protoc -I=. -I=$GOPATH/src -I=$GOPATH/src/github.com/gogo/protobuf/protobuf --{binary}_out=. myproto.proto

To use proto files from "google/protobuf" you need to add additional args to protoc.

    protoc -I=. -I=$GOPATH/src -I=$GOPATH/src/github.com/gogo/protobuf/protobuf --{binary}_out=\
    Mgoogle/protobuf/any.proto=github.com/gogo/protobuf/types,\
    Mgoogle/protobuf/duration.proto=github.com/gogo/protobuf/types,\
    Mgoogle/protobuf/struct.proto=github.com/gogo/protobuf/types,\
    Mgoogle/protobuf/timestamp.proto=github.com/gogo/protobuf/types,\
    Mgoogle/protobuf/wrappers.proto=github.com/gogo/protobuf/types:. \
    myproto.proto

Note that in the protoc command, {binary} does not contain the initial prefix of "protoc-gen".

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

See [https://github.com/gogo/grpc-example](https://github.com/gogo/grpc-example) for an example of using gRPC with gogoprotobuf and the wider grpc-ecosystem.


## License
This software is licensed under the 3-Clause BSD License
("BSD License 2.0", "Revised BSD License", "New BSD License", or "Modified BSD License").


