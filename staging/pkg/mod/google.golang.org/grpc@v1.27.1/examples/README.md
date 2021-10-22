gRPC in 3 minutes (Go)
======================

BACKGROUND
-------------
For this sample, we've already generated the server and client stubs from [helloworld.proto](helloworld/helloworld/helloworld.proto).

PREREQUISITES
-------------

- This requires Go 1.9 or later
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

1. Install [protobuf compiler](https://github.com/google/protobuf/blob/master/README.md#protocol-compiler-installation)

1. Install the protoc Go plugin

   ```
   $ go get -u github.com/golang/protobuf/protoc-gen-go
   ```

1. Rebuild the generated Go code

   ```
   $ go generate google.golang.org/grpc/examples/helloworld/...
   ```

   Or run `protoc` command (with the grpc plugin)

   ```
   $ protoc -I helloworld/ helloworld/helloworld.proto --go_out=plugins=grpc:helloworld
   ```
