# Download and build

## System requirements

The etcd performance benchmarks run etcd on 8 vCPU, 16GB RAM, 50GB SSD GCE instances, but any relatively modern machine with low latency storage and a few gigabytes of memory should suffice for most use cases. Applications with large v2 data stores will require more memory than a large v3 data store since data is kept in anonymous memory instead of memory mapped from a file. than For running etcd on a cloud provider, we suggest at least a medium instance on AWS or a standard-1 instance on GCE.

## Download the pre-built binary

The easiest way to get etcd is to use one of the pre-built release binaries which are available for OSX, Linux, Windows, appc, and Docker. Instructions for using these binaries are on the [GitHub releases page][github-release].

## Build the latest version

For those wanting to try the very latest version, build etcd from the `master` branch. [Go](https://golang.org/) version 1.7+ is required to build the latest version of etcd. To ensure etcd is built against well-tested libraries, etcd vendors its dependencies for official release binaries. However, etcd's vendoring is also optional to avoid potential import conflicts when embedding the etcd server or using the etcd client.

First, confirm go 1.7+ is installed:

```sh
# go is required
$ go version
go version go1.7.3 darwin/amd64

```

To build `etcd` from the `master` branch without a `GOPATH` using the official `build` script:

```sh
$ git clone https://github.com/coreos/etcd.git
$ cd etcd
$ ./build
$ ./bin/etcd
```

To build a vendored `etcd` from the `master` branch via `go get`:

```sh
# GOPATH should be set
$ echo $GOPATH
/Users/example/go
$ go get github.com/coreos/etcd/cmd/etcd
$ $GOPATH/bin/etcd
```

To build `etcd` from the `master` branch without vendoring (may not build due to upstream conflicts):

```sh
# GOPATH should be set
$ echo $GOPATH
/Users/example/go
$ go get github.com/coreos/etcd
$ $GOPATH/bin/etcd
```

## Test the installation

Check the etcd binary is built correctly by starting etcd and setting a key.

Start etcd:

```
$ ./bin/etcd
```

Set a key:

```
$ ETCDCTL_API=3 ./bin/etcdctl put foo bar
OK
```

If OK is printed, then etcd is working!

[github-release]: https://github.com/coreos/etcd/releases/
[go]: https://golang.org/doc/install
[build-script]: ../build
[cmd-directory]: ../cmd

