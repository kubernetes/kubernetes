// Package proxyprotocol implement receiver for HA Proxy Protocol V1 and V2.
//
// Proxy Protocol spec http://www.haproxy.org/download/2.0/doc/proxy-protocol.txt
//
// This package provides a wrapper for the interface net.Listener, which extracts
// remote and local address of the connection from the headers in the format
// HA proxy protocol.
package proxyprotocol

import (
	"bufio"
	"errors"
	"net"
)

// Header struct represent header parsing result
type Header struct {
	SrcAddr net.Addr
	DstAddr net.Addr
}

// HeaderParserBuilder build HeaderParser's
type HeaderParserBuilder interface {
	Build(Logger) HeaderParser
}

// HeaderParser describe interface for header parsers
type HeaderParser interface {
	Parse(readBuf *bufio.Reader) (*Header, error)
}

// Shared HeaderParser errors
var (
	ErrInvalidSignature = errors.New("invalid signature")
	ErrUnknownProtocol  = errors.New("unknown protocol")
)
