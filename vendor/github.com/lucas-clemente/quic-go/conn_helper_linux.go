//go:build linux
// +build linux

package quic

import "golang.org/x/sys/unix"

const (
	msgTypeIPTOS            = unix.IP_TOS
	disablePathMTUDiscovery = false
)

const (
	ipv4RECVPKTINFO = unix.IP_PKTINFO
	ipv6RECVPKTINFO = unix.IPV6_RECVPKTINFO
)

const (
	msgTypeIPv4PKTINFO = unix.IP_PKTINFO
	msgTypeIPv6PKTINFO = unix.IPV6_PKTINFO
)

const batchSize = 8 // needs to smaller than MaxUint8 (otherwise the type of oobConn.readPos has to be changed)
