# Metadata

gRPC supports sending metadata between client and server.
This doc shows how to send and receive metadata in gRPC-go.

## Background

Four kinds of service method:

- [Unary RPC](http://www.grpc.io/docs/guides/concepts.html#unary-rpc)
- [Server streaming RPC](http://www.grpc.io/docs/guides/concepts.html#server-streaming-rpc)
- [Client streaming RPC](http://www.grpc.io/docs/guides/concepts.html#client-streaming-rpc)
- [Bidirectional streaming RPC](http://www.grpc.io/docs/guides/concepts.html#bidirectional-streaming-rpc)

And concept of [metadata](http://www.grpc.io/docs/guides/concepts.html#metadata).

## Constructing metadata

A metadata can be created using package [metadata](https://godoc.org/google.golang.org/grpc/metadata).
The type MD is actually a map from string to a list of strings:

```go
type MD map[string][]string
```

Metadata can be read like a normal map.
Note that the value type of this map is `[]string`,
so that users can attach multiple values using a single key.

### Creating a new metadata

A metadata can be created from a `map[string]string` using function `New`:

```go
md := metadata.New(map[string]string{"key1": "val1", "key2": "val2"})
```

Another way is to use `Pairs`.
Values with the same key will be merged into a list:

```go
md := metadata.Pairs(
    "key1", "val1",
    "key1", "val1-2", // "key1" will have map value []string{"val1", "val1-2"}
    "key2", "val2",
)
```

__Note:__ all the keys will be automatically converted to lowercase,
so "key1" and "kEy1" will be the same key and their values will be merged into the same list.
This happens for both `New` and `Pairs`.

### Storing binary data in metadata

In metadata, keys are always strings. But values can be strings or binary data.
To store binary data value in metadata, simply add "-bin" suffix to the key.
The values with "-bin" suffixed keys will be encoded when creating the metadata:

```go
md := metadata.Pairs(
    "key", "string value",
    "key-bin", string([]byte{96, 102}), // this binary data will be encoded (base64) before sending
                                        // and will be decoded after being transferred.
)
```

## Retrieving metadata from context

Metadata can be retrieved from context using `FromIncomingContext`:

```go
func (s *server) SomeRPC(ctx context.Context, in *pb.SomeRequest) (*pb.SomeResponse, err) {
    md, ok := metadata.FromIncomingContext(ctx)
    // do something with metadata
}
```

## Sending and receiving metadata - client side

[//]: # "TODO: uncomment next line after example source added"
[//]: # "Real metadata sending and receiving examples are available [here](TODO:example_dir)."

### Sending metadata

To send metadata to server, the client can wrap the metadata into a context using `NewContext`, and make the RPC with this context:

```go
md := metadata.Pairs("key", "val")

// create a new context with this metadata
ctx := metadata.NewOutgoingContext(context.Background(), md)

// make unary RPC
response, err := client.SomeRPC(ctx, someRequest)

// or make streaming RPC
stream, err := client.SomeStreamingRPC(ctx)
```

To read this back from the context on the client (e.g. in an interceptor) before the RPC is sent, use `FromOutgoingContext`.

### Receiving metadata

Metadata that a client can receive includes header and trailer.

#### Unary call

Header and trailer sent along with a unary call can be retrieved using function [Header](https://godoc.org/google.golang.org/grpc#Header) and [Trailer](https://godoc.org/google.golang.org/grpc#Trailer) in [CallOption](https://godoc.org/google.golang.org/grpc#CallOption):

```go
var header, trailer metadata.MD // variable to store header and trailer
r, err := client.SomeRPC(
    ctx,
    someRequest,
    grpc.Header(&header),    // will retrieve header
    grpc.Trailer(&trailer),  // will retrieve trailer
)

// do something with header and trailer
```

#### Streaming call

For streaming calls including:

- Server streaming RPC
- Client streaming RPC
- Bidirectional streaming RPC

Header and trailer can be retrieved from the returned stream using function `Header` and `Trailer` in interface [ClientStream](https://godoc.org/google.golang.org/grpc#ClientStream):

```go
stream, err := client.SomeStreamingRPC(ctx)

// retrieve header
header, err := stream.Header()

// retrieve trailer
trailer := stream.Trailer()

```

## Sending and receiving metadata - server side

[//]: # "TODO: uncomment next line after example source added"
[//]: # "Real metadata sending and receiving examples are available [here](TODO:example_dir)."

### Receiving metadata

To read metadata sent by the client, the server needs to retrieve it from RPC context.
If it is a unary call, the RPC handler's context can be used.
For streaming calls, the server needs to get context from the stream.

#### Unary call

```go
func (s *server) SomeRPC(ctx context.Context, in *pb.someRequest) (*pb.someResponse, error) {
    md, ok := metadata.FromIncomingContext(ctx)
    // do something with metadata
}
```

#### Streaming call

```go
func (s *server) SomeStreamingRPC(stream pb.Service_SomeStreamingRPCServer) error {
    md, ok := metadata.FromIncomingContext(stream.Context()) // get context from stream
    // do something with metadata
}
```

### Sending metadata

#### Unary call

To send header and trailer to client in unary call, the server can call [SendHeader](https://godoc.org/google.golang.org/grpc#SendHeader) and [SetTrailer](https://godoc.org/google.golang.org/grpc#SetTrailer) functions in module [grpc](https://godoc.org/google.golang.org/grpc).
These two functions take a context as the first parameter.
It should be the RPC handler's context or one derived from it:

```go
func (s *server) SomeRPC(ctx context.Context, in *pb.someRequest) (*pb.someResponse, error) {
    // create and send header
    header := metadata.Pairs("header-key", "val")
    grpc.SendHeader(ctx, header)
    // create and set trailer
    trailer := metadata.Pairs("trailer-key", "val")
    grpc.SetTrailer(ctx, trailer)
}
```

#### Streaming call

For streaming calls, header and trailer can be sent using function `SendHeader` and `SetTrailer` in interface [ServerStream](https://godoc.org/google.golang.org/grpc#ServerStream):

```go
func (s *server) SomeStreamingRPC(stream pb.Service_SomeStreamingRPCServer) error {
    // create and send header
    header := metadata.Pairs("header-key", "val")
    stream.SendHeader(header)
    // create and set trailer
    trailer := metadata.Pairs("trailer-key", "val")
    stream.SetTrailer(trailer)
}
```
