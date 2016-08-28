package types

import (
	"io"
	"strings"
)

// EndpointType is a type of endpoint.
type EndpointType int

const (
	// UnknownEndpointType is an unknown endpoint type.
	UnknownEndpointType EndpointType = iota

	// UnixEndpoint is a UNIX socket endpoint.
	UnixEndpoint

	// TCPEndpoint is a TCP endpoint.
	TCPEndpoint
)

// String returns the endpoint type's string representation.
func (t EndpointType) String() string {
	switch t {
	case UnixEndpoint:
		return "unix"
	case TCPEndpoint:
		return "tcp"
	default:
		return ""
	}
}

// ParseEndpointType parses the endpoint type.
func ParseEndpointType(str string) EndpointType {
	str = strings.ToLower(str)
	switch str {
	case "unix":
		return UnixEndpoint
	case "tcp":
		return TCPEndpoint
	}
	return UnknownEndpointType
}

// Server is the interface for a libStorage server.
type Server interface {
	io.Closer

	// Name returns the name of the server.
	Name() string

	// Addrs returns the server's configured endpoint addresses.
	Addrs() []string
}
