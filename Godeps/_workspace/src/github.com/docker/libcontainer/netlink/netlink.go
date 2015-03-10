// Packet netlink provide access to low level Netlink sockets and messages.
//
// Actual implementations are in:
// netlink_linux.go
// netlink_darwin.go
package netlink

import (
	"errors"
	"net"
)

var (
	ErrWrongSockType   = errors.New("Wrong socket type")
	ErrShortResponse   = errors.New("Got short response from netlink")
	ErrInterfaceExists = errors.New("Network interface already exists")
)

// A Route is a subnet associated with the interface to reach it.
type Route struct {
	*net.IPNet
	Iface   *net.Interface
	Default bool
}

// An IfAddr defines IP network settings for a given network interface
type IfAddr struct {
	Iface *net.Interface
	IP    net.IP
	IPNet *net.IPNet
}
