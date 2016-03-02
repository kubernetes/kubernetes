gRPC in 3 minutes (Go)
======================

BACKGROUND
-------------
For this sample, we've already generated the server and client stubs from [helloworld.proto](helloworld/helloworld/helloworld.proto).

PREREQUISITES
-------------

- This requires Go 1.4
- Requires that [GOPATH is set](https://golang.org/doc/code.html#GOPATH)

```
$ go help gopath
$ # ensure the PATH contains $GOPATH/bin
$ export PATH=$PATH:$GOPATH/bin
```

INSTALL
-------

```
$ go get -u google.golang.org/grpc/examples/helloworld/greeter_client
$ go get -u google.golang.org/grpc/examples/helloworld/greeter_server
```

TRY IT!
-------

- Run the server

```
$ greeter_server &
```

- Run the client

```
$ greeter_client
```

OPTIONAL - Rebuilding the generated code
----------------------------------------

1 First [install protoc](https://github.com/google/protobuf/blob/master/README.md)
  - For now, this needs to be installed from source
  - This is will change once proto3 is officially released

2 Install the protoc Go plugin.

```
$ go get -a github.com/golang/protobuf/protoc-gen-go
$
$ # from this dir; invoke protoc
$  protoc -I ./helloworld/helloworld/ ./helloworld/helloworld/helloworld.proto --go_out=plugins=grpc:helloworld
```
