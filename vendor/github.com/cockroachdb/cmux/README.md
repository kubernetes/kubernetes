# cmux: Connection Mux [![Build Status](https://travis-ci.org/cockroachdb/cmux.svg?branch=master)](https://travis-ci.org/cockroachdb/cmux) [![GoDoc](https://godoc.org/github.com/cockroachdb/cmux?status.svg)](https://godoc.org/github.com/cockroachdb/cmux)

cmux is a generic Go library to multiplex connections based on their payload.
Using cmux, you can serve gRPC, SSH, HTTPS, HTTP, Go RPC, and pretty much any
other protocol on the same TCP listener.

## How-To
Simply create your main listener, create a cmux for that listener,
and then match connections:
```go
// Create the main listener.
l, err := net.Listen("tcp", ":23456")
if err != nil {
	log.Fatal(err)
}

// Create a cmux.
m := cmux.New(l)

// Match connections in order:
// First grpc, then HTTP, and otherwise Go RPC/TCP.
grpcL := m.Match(cmux.HTTP2HeaderField("content-type", "application/grpc"))
httpL := m.Match(cmux.HTTP1Fast())
trpcL := m.Match(cmux.Any()) // Any means anything that is not yet matched.

// Create your protocol servers.
grpcS := grpc.NewServer()
grpchello.RegisterGreeterServer(grpcs, &server{})

httpS := &http.Server{
	Handler: &helloHTTP1Handler{},
}

trpcS := rpc.NewServer()
s.Register(&ExampleRPCRcvr{})

// Use the muxed listeners for your servers.
go grpcS.Serve(grpcL)
go httpS.Serve(httpL)
go trpcS.Accept(trpcL)

// Start serving!
m.Serve()
```

There are [more examples on GoDoc](https://godoc.org/github.com/cockroachdb/cmux#pkg-examples).

## Performance
Since we are only matching the very first bytes of a connection, the
performance overhead on long-lived connections (i.e., RPCs and pipelined HTTP
streams) is negligible.

## Limitations
* *TLS*: `net/http` uses a [type assertion](https://github.com/golang/go/issues/14221)
to identify TLS connections; since cmux's lookahead-implementing connection
wraps the underlying TLS connection, this type assertion fails. This means you
can serve HTTPS using cmux but `http.Request.TLS` will not be set in your
handlers. If you are able to wrap TLS around cmux, you can work around this
limitation. See https://github.com/cockroachdb/cockroach/commit/83caba2 for an
example of this approach.

* *Different Protocols on The Same Connection*: `cmux` matches the connection
when it's accepted. For example, one connection can be either gRPC or REST, but
not both. That is, we assume that a client connection is either used for gRPC
or REST.
