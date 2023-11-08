# go-proxyprotocol

[![GoDoc](https://godoc.org/github.com/c0va23/go-proxyprotocol?status.svg)](https://godoc.org/github.com/c0va23/go-proxyprotocol)
[![Go Report Card](https://goreportcard.com/badge/github.com/c0va23/go-proxyprotocol)](https://goreportcard.com/report/github.com/c0va23/go-proxyprotocol)
[![Build Status](https://travis-ci.org/c0va23/go-proxyprotocol.svg?branch=master)](https://travis-ci.org/c0va23/go-proxyprotocol)

Golang package `github.com/c0va23/go-proxyprotocol' provide receiver for
[HA ProxyProtocol v1 and v2](http://www.haproxy.org/download/2.0/doc/proxy-protocol.txt).

This package provides a wrapper for the interface net.Listener, which extracts
remote and local address of the connection from the headers in the format
HA proxyprotocol.

## Usage example

```go
package main

import (
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/c0va23/go-proxyprotocol"
)

func main() {
	rawList, _ := net.Listen("tcp", ":8080")

	list := proxyprotocol.
        NewDefaultListener(rawList).
        WithLogger(proxyprotocol.LoggerFunc(log.Printf))

	http.Serve(list, http.HandlerFunc(func(res http.ResponseWriter, req *http.Request) {
		log.Printf("Remote Addr: %s, URI: %s", req.RemoteAddr, req.RequestURI)
		fmt.Fprintf(res, "Hello, %s!\n", req.RemoteAddr)
	}))
}
```

DefaultListener try parse proxyprotocol v1 and v2 header. If header signature
not recognized, then used raw connection.

If you want to use only proxy protocol V1 or v2 headers, you can initialize the
listener as follows:

```go
list := proxyprotocol.NewListener(rawList, proxyprotocol.TextHeaderParserBuilder)
```

## Implementation status

### Human-readable header format (Version 1)
- [x] UNKNOWN
- [x] IPv4
- [x] IPv6

### Binary header format (version 2)
- [x] Unspec
- [x] TCP over IPv4
- [x] TCP over IPv6
- [ ] UDP over IPv4
- [ ] UDP over IPv6
- [ ] Unix Stream
- [ ] Unix Datagram
