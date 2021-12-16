//go:build !darwin && !linux && !freebsd && !windows
// +build !darwin,!linux,!freebsd,!windows

package quic

import "net"

const disablePathMTUDiscovery = false

func newConn(c net.PacketConn) (connection, error) {
	return &basicConn{PacketConn: c}, nil
}

func inspectReadBuffer(interface{}) (int, error) {
	return 0, nil
}

func (i *packetInfo) OOB() []byte { return nil }
