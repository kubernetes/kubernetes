// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_dragonfly.go

// +build dragonfly

package ipv6

const (
	sysIPV6_UNICAST_HOPS   = 0x4
	sysIPV6_MULTICAST_IF   = 0x9
	sysIPV6_MULTICAST_HOPS = 0xa
	sysIPV6_MULTICAST_LOOP = 0xb
	sysIPV6_JOIN_GROUP     = 0xc
	sysIPV6_LEAVE_GROUP    = 0xd
	sysIPV6_PORTRANGE      = 0xe
	sysICMP6_FILTER        = 0x12

	sysIPV6_CHECKSUM = 0x1a
	sysIPV6_V6ONLY   = 0x1b

	sysIPV6_IPSEC_POLICY = 0x1c

	sysIPV6_RTHDRDSTOPTS = 0x23
	sysIPV6_RECVPKTINFO  = 0x24
	sysIPV6_RECVHOPLIMIT = 0x25
	sysIPV6_RECVRTHDR    = 0x26
	sysIPV6_RECVHOPOPTS  = 0x27
	sysIPV6_RECVDSTOPTS  = 0x28

	sysIPV6_USE_MIN_MTU = 0x2a
	sysIPV6_RECVPATHMTU = 0x2b

	sysIPV6_PATHMTU = 0x2c

	sysIPV6_PKTINFO  = 0x2e
	sysIPV6_HOPLIMIT = 0x2f
	sysIPV6_NEXTHOP  = 0x30
	sysIPV6_HOPOPTS  = 0x31
	sysIPV6_DSTOPTS  = 0x32
	sysIPV6_RTHDR    = 0x33

	sysIPV6_RECVTCLASS = 0x39

	sysIPV6_AUTOFLOWLABEL = 0x3b

	sysIPV6_TCLASS   = 0x3d
	sysIPV6_DONTFRAG = 0x3e

	sysIPV6_PREFER_TEMPADDR = 0x3f

	sysIPV6_PORTRANGE_DEFAULT = 0x0
	sysIPV6_PORTRANGE_HIGH    = 0x1
	sysIPV6_PORTRANGE_LOW     = 0x2

	sysSizeofSockaddrInet6 = 0x1c
	sysSizeofInet6Pktinfo  = 0x14
	sysSizeofIPv6Mtuinfo   = 0x20

	sysSizeofIPv6Mreq = 0x14

	sysSizeofICMPv6Filter = 0x20
)

type sysSockaddrInet6 struct {
	Len      uint8
	Family   uint8
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte /* in6_addr */
	Scope_id uint32
}

type sysInet6Pktinfo struct {
	Addr    [16]byte /* in6_addr */
	Ifindex uint32
}

type sysIPv6Mtuinfo struct {
	Addr sysSockaddrInet6
	Mtu  uint32
}

type sysIPv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Interface uint32
}

type sysICMPv6Filter struct {
	Filt [8]uint32
}
