//go:build darwin
// +build darwin

package quic

import "golang.org/x/sys/unix"

const (
	msgTypeIPTOS            = unix.IP_RECVTOS
	disablePathMTUDiscovery = false
)

const (
	ipv4RECVPKTINFO = unix.IP_RECVPKTINFO
	ipv6RECVPKTINFO = 0x3d
)

const (
	msgTypeIPv4PKTINFO = unix.IP_PKTINFO
	msgTypeIPv6PKTINFO = 0x2e
)

// ReadBatch only returns a single packet on OSX,
// see https://godoc.org/golang.org/x/net/ipv4#PacketConn.ReadBatch.
const batchSize = 1
