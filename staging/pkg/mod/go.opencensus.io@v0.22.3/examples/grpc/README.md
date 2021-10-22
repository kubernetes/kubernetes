# Example gRPC server and client with OpenCensus

This example uses:

* gRPC to create an RPC server and client.
* The OpenCensus gRPC plugin to instrument the RPC server and client.
* Debugging exporters to print stats and traces to stdout.

```
$ go get go.opencensus.io/examples/grpc/...
```

First, run the server:

```
$ go run $(go env GOPATH)/src/go.opencensus.io/examples/grpc/helloworld_server/main.go
```

Then, run the client:

```
$ go run $(go env GOPATH)/src/go.opencensus.io/examples/grpc/helloworld_client/main.go
```

You will see traces and stats exported on the stdout. You can use one of the
[exporters](https://godoc.org/go.opencensus.io/exporter)
to upload collected data to the backend of your choice.

You can also see the z-pages provided from the server:
* Traces: http://localhost:8081/debug/tracez
* RPCs: http://localhost:8081/debug/rpcz
