# Yamux

Yamux (Yet another Multiplexer) is a multiplexing library for Golang.
It relies on an underlying connection to provide reliability
and ordering, such as TCP or Unix domain sockets, and provides
stream-oriented multiplexing. It is inspired by SPDY but is not
interoperable with it.

Yamux features include:

* Bi-directional streams
  * Streams can be opened by either client or server
  * Useful for NAT traversal
  * Server-side push support
* Flow control
  * Avoid starvation
  * Back-pressure to prevent overwhelming a receiver
* Keep Alives
  * Enables persistent connections over a load balancer
* Efficient
  * Enables thousands of logical streams with low overhead

## Documentation

For complete documentation, see the associated [Godoc](http://godoc.org/github.com/hashicorp/yamux).

## Specification

The full specification for Yamux is provided in the `spec.md` file.
It can be used as a guide to implementors of interoperable libraries.

## Usage

Using Yamux is remarkably simple:

```go

func client() {
    // Get a TCP connection
    conn, err := net.Dial(...)
    if err != nil {
        panic(err)
    }

    // Setup client side of yamux
    session, err := yamux.Client(conn, nil)
    if err != nil {
        panic(err)
    }

    // Open a new stream
    stream, err := session.Open()
    if err != nil {
        panic(err)
    }

    // Stream implements net.Conn
    stream.Write([]byte("ping"))
}

func server() {
    // Accept a TCP connection
    conn, err := listener.Accept()
    if err != nil {
        panic(err)
    }

    // Setup server side of yamux
    session, err := yamux.Server(conn, nil)
    if err != nil {
        panic(err)
    }

    // Accept a stream
    stream, err := session.Accept()
    if err != nil {
        panic(err)
    }

    // Listen for a message
    buf := make([]byte, 4)
    stream.Read(buf)
}

```

