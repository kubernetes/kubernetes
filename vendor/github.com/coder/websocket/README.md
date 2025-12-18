# websocket

[![Go Reference](https://pkg.go.dev/badge/github.com/coder/websocket.svg)](https://pkg.go.dev/github.com/coder/websocket)
[![Go Coverage](https://coder.github.io/websocket/coverage.svg)](https://coder.github.io/websocket/coverage.html)

websocket is a minimal and idiomatic WebSocket library for Go.

## Install

```sh
go get github.com/coder/websocket
```

> [!NOTE]
> Coder now maintains this project as explained in [this blog post](https://coder.com/blog/websocket).
> We're grateful to [nhooyr](https://github.com/nhooyr) for authoring and maintaining this project from
> 2019 to 2024.

## Highlights

- Minimal and idiomatic API
- First class [context.Context](https://blog.golang.org/context) support
- Fully passes the WebSocket [autobahn-testsuite](https://github.com/crossbario/autobahn-testsuite)
- [Zero dependencies](https://pkg.go.dev/github.com/coder/websocket?tab=imports)
- JSON helpers in the [wsjson](https://pkg.go.dev/github.com/coder/websocket/wsjson) subpackage
- Zero alloc reads and writes
- Concurrent writes
- [Close handshake](https://pkg.go.dev/github.com/coder/websocket#Conn.Close)
- [net.Conn](https://pkg.go.dev/github.com/coder/websocket#NetConn) wrapper
- [Ping pong](https://pkg.go.dev/github.com/coder/websocket#Conn.Ping) API
- [RFC 7692](https://tools.ietf.org/html/rfc7692) permessage-deflate compression
- [CloseRead](https://pkg.go.dev/github.com/coder/websocket#Conn.CloseRead) helper for write only connections
- Compile to [Wasm](https://pkg.go.dev/github.com/coder/websocket#hdr-Wasm)

## Roadmap

See GitHub issues for minor issues but the major future enhancements are:

- [ ] Perfect examples [#217](https://github.com/nhooyr/websocket/issues/217)
- [ ] wstest.Pipe for in memory testing [#340](https://github.com/nhooyr/websocket/issues/340)
- [ ] Ping pong heartbeat helper [#267](https://github.com/nhooyr/websocket/issues/267)
- [ ] Ping pong instrumentation callbacks [#246](https://github.com/nhooyr/websocket/issues/246)
- [ ] Graceful shutdown helpers [#209](https://github.com/nhooyr/websocket/issues/209)
- [ ] Assembly for WebSocket masking [#16](https://github.com/nhooyr/websocket/issues/16)
  - WIP at [#326](https://github.com/nhooyr/websocket/pull/326), about 3x faster
- [ ] HTTP/2 [#4](https://github.com/nhooyr/websocket/issues/4)
- [ ] The holy grail [#402](https://github.com/nhooyr/websocket/issues/402)

## Examples

For a production quality example that demonstrates the complete API, see the
[echo example](./internal/examples/echo).

For a full stack example, see the [chat example](./internal/examples/chat).

### Server

```go
http.HandlerFunc(func (w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		// ...
	}
	defer c.CloseNow()

	// Set the context as needed. Use of r.Context() is not recommended
	// to avoid surprising behavior (see http.Hijacker).
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	var v any
	err = wsjson.Read(ctx, c, &v)
	if err != nil {
		// ...
	}

	log.Printf("received: %v", v)

	c.Close(websocket.StatusNormalClosure, "")
})
```

### Client

```go
ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
defer cancel()

c, _, err := websocket.Dial(ctx, "ws://localhost:8080", nil)
if err != nil {
	// ...
}
defer c.CloseNow()

err = wsjson.Write(ctx, c, "hi")
if err != nil {
	// ...
}

c.Close(websocket.StatusNormalClosure, "")
```

## Comparison

### gorilla/websocket

Advantages of [gorilla/websocket](https://github.com/gorilla/websocket):

- Mature and widely used
- [Prepared writes](https://pkg.go.dev/github.com/gorilla/websocket#PreparedMessage)
- Configurable [buffer sizes](https://pkg.go.dev/github.com/gorilla/websocket#hdr-Buffers)
- No extra goroutine per connection to support cancellation with context.Context. This costs github.com/coder/websocket 2 KB of memory per connection.
  - Will be removed soon with [context.AfterFunc](https://github.com/golang/go/issues/57928). See [#411](https://github.com/nhooyr/websocket/issues/411)

Advantages of github.com/coder/websocket:

- Minimal and idiomatic API
  - Compare godoc of [github.com/coder/websocket](https://pkg.go.dev/github.com/coder/websocket) with [gorilla/websocket](https://pkg.go.dev/github.com/gorilla/websocket) side by side.
- [net.Conn](https://pkg.go.dev/github.com/coder/websocket#NetConn) wrapper
- Zero alloc reads and writes ([gorilla/websocket#535](https://github.com/gorilla/websocket/issues/535))
- Full [context.Context](https://blog.golang.org/context) support
- Dial uses [net/http.Client](https://golang.org/pkg/net/http/#Client)
  - Will enable easy HTTP/2 support in the future
  - Gorilla writes directly to a net.Conn and so duplicates features of net/http.Client.
- Concurrent writes
- Close handshake ([gorilla/websocket#448](https://github.com/gorilla/websocket/issues/448))
- Idiomatic [ping pong](https://pkg.go.dev/github.com/coder/websocket#Conn.Ping) API
  - Gorilla requires registering a pong callback before sending a Ping
- Can target Wasm ([gorilla/websocket#432](https://github.com/gorilla/websocket/issues/432))
- Transparent message buffer reuse with [wsjson](https://pkg.go.dev/github.com/coder/websocket/wsjson) subpackage
- [1.75x](https://github.com/nhooyr/websocket/releases/tag/v1.7.4) faster WebSocket masking implementation in pure Go
  - Gorilla's implementation is slower and uses [unsafe](https://golang.org/pkg/unsafe/).
    Soon we'll have assembly and be 3x faster [#326](https://github.com/nhooyr/websocket/pull/326)
- Full [permessage-deflate](https://tools.ietf.org/html/rfc7692) compression extension support
  - Gorilla only supports no context takeover mode
- [CloseRead](https://pkg.go.dev/github.com/coder/websocket#Conn.CloseRead) helper for write only connections ([gorilla/websocket#492](https://github.com/gorilla/websocket/issues/492))

#### golang.org/x/net/websocket

[golang.org/x/net/websocket](https://pkg.go.dev/golang.org/x/net/websocket) is deprecated.
See [golang/go/issues/18152](https://github.com/golang/go/issues/18152).

The [net.Conn](https://pkg.go.dev/github.com/coder/websocket#NetConn) can help in transitioning
to github.com/coder/websocket.

#### gobwas/ws

[gobwas/ws](https://github.com/gobwas/ws) has an extremely flexible API that allows it to be used
in an event driven style for performance. See the author's [blog post](https://medium.freecodecamp.org/million-websockets-and-go-cc58418460bb).

However it is quite bloated. See https://pkg.go.dev/github.com/gobwas/ws

When writing idiomatic Go, github.com/coder/websocket will be faster and easier to use.

#### lesismal/nbio

[lesismal/nbio](https://github.com/lesismal/nbio) is similar to gobwas/ws in that the API is
event driven for performance reasons.

However it is quite bloated. See https://pkg.go.dev/github.com/lesismal/nbio

When writing idiomatic Go, github.com/coder/websocket will be faster and easier to use.
