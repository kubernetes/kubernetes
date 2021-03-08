# go-proxyproto

[![Actions Status](https://github.com/pires/go-proxyproto/workflows/test/badge.svg)](https://github.com/pires/go-proxyproto/actions)
[![Coverage Status](https://coveralls.io/repos/github/pires/go-proxyproto/badge.svg?branch=master)](https://coveralls.io/github/pires/go-proxyproto?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/pires/go-proxyproto)](https://goreportcard.com/report/github.com/pires/go-proxyproto)
[![](https://godoc.org/github.com/pires/go-proxyproto?status.svg)](https://pkg.go.dev/github.com/pires/go-proxyproto?tab=doc)


A Go library implementation of the [PROXY protocol, versions 1 and 2](https://www.haproxy.org/download/2.3/doc/proxy-protocol.txt),
which provides, as per specification:
> (...) a convenient way to safely transport connection
> information such as a client's address across multiple layers of NAT or TCP
> proxies. It is designed to require little changes to existing components and
> to limit the performance impact caused by the processing of the transported
> information.

This library is to be used in one of or both proxy clients and proxy servers that need to support said protocol.
Both protocol versions, 1 (text-based) and 2 (binary-based) are supported.

## Installation

```shell
$ go get -u github.com/pires/go-proxyproto
```

## Usage

### Client

```go
package main

import (
	"io"
	"log"
	"net"

	proxyproto "github.com/pires/go-proxyproto"
)

func chkErr(err error) {
	if err != nil {
		log.Fatalf("Error: %s", err.Error())
	}
}

func main() {
	// Dial some proxy listener e.g. https://github.com/mailgun/proxyproto
	target, err := net.ResolveTCPAddr("tcp", "127.0.0.1:2319")
	chkErr(err)

	conn, err := net.DialTCP("tcp", nil, target)
	chkErr(err)

	defer conn.Close()

	// Create a proxyprotocol header or use HeaderProxyFromAddrs() if you
	// have two conn's
	header := &proxyproto.Header{
		Version:            1,
		Command:            proxyproto.PROXY,
		TransportProtocol:  proxyproto.TCPv4,
		SourceAddr: &net.TCPAddr{
			IP:   net.ParseIP("10.1.1.1"),
			Port: 1000,
		},
		DestinationAddr: &net.TCPAddr{
			IP:   net.ParseIP("20.2.2.2"),
			Port: 2000,
		},
	}
	// After the connection was created write the proxy headers first
	_, err = header.WriteTo(conn)
	chkErr(err)
	// Then your data... e.g.:
	_, err = io.WriteString(conn, "HELO")
	chkErr(err)
}
```

### Server

```go
package main

import (
	"log"
	"net"

	proxyproto "github.com/pires/go-proxyproto"
)

func main() {
	// Create a listener
	addr := "localhost:9876"
	list, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("couldn't listen to %q: %q\n", addr, err.Error())
	}

	// Wrap listener in a proxyproto listener
	proxyListener := &proxyproto.Listener{Listener: list}
	defer proxyListener.Close()

	// Wait for a connection and accept it
	conn, err := proxyListener.Accept()
	defer conn.Close()

	// Print connection details
	if conn.LocalAddr() == nil {
		log.Fatal("couldn't retrieve local address")
	}
	log.Printf("local address: %q", conn.LocalAddr().String())

	if conn.RemoteAddr() == nil {
		log.Fatal("couldn't retrieve remote address")
	}
	log.Printf("remote address: %q", conn.RemoteAddr().String())
}
```

## Special notes

### AWS

AWS Network Load Balancer (NLB) does not push the PPV2 header until the client starts sending the data. This is a problem if your server speaks first. e.g. SMTP, FTP, SSH etc.

By default, NLB target group attribute `proxy_protocol_v2.client_to_server.header_placement` has the value `on_first_ack_with_payload`. You need to contact AWS support to change it to `on_first_ack`, instead.

Just to be clear, you need this fix only if your server is designed to speak first.
