// +build !linux,!plan9

package dns

import (
	"net"
	"syscall"
)

// These do nothing. See udp_linux.go for an example of how to implement this.

// We tried to adhire to some kind of naming scheme.

func setUDPSocketOptions4(conn *net.UDPConn) error                 { return nil }
func setUDPSocketOptions6(conn *net.UDPConn) error                 { return nil }
func getUDPSocketOptions6Only(conn *net.UDPConn) (bool, error)     { return false, nil }
func getUDPSocketName(conn *net.UDPConn) (syscall.Sockaddr, error) { return nil, nil }
