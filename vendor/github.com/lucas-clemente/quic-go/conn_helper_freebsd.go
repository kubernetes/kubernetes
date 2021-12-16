//go:build freebsd
// +build freebsd

package quic

import "golang.org/x/sys/unix"

const (
	msgTypeIPTOS            = unix.IP_RECVTOS
	disablePathMTUDiscovery = false
)

const (
	ipv4RECVPKTINFO = 0x7
	ipv6RECVPKTINFO = 0x24
)

const (
	msgTypeIPv4PKTINFO = 0x7
	msgTypeIPv6PKTINFO = 0x2e
)

const batchSize = 8
