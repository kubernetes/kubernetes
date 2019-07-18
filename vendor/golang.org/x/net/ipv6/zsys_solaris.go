// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_solaris.go

package ipv6

const (
	sysIPV6_UNICAST_HOPS   = 0x5
	sysIPV6_MULTICAST_IF   = 0x6
	sysIPV6_MULTICAST_HOPS = 0x7
	sysIPV6_MULTICAST_LOOP = 0x8
	sysIPV6_JOIN_GROUP     = 0x9
	sysIPV6_LEAVE_GROUP    = 0xa

	sysIPV6_PKTINFO = 0xb

	sysIPV6_HOPLIMIT = 0xc
	sysIPV6_NEXTHOP  = 0xd
	sysIPV6_HOPOPTS  = 0xe
	sysIPV6_DSTOPTS  = 0xf

	sysIPV6_RTHDR        = 0x10
	sysIPV6_RTHDRDSTOPTS = 0x11

	sysIPV6_RECVPKTINFO  = 0x12
	sysIPV6_RECVHOPLIMIT = 0x13
	sysIPV6_RECVHOPOPTS  = 0x14

	sysIPV6_RECVRTHDR = 0x16

	sysIPV6_RECVRTHDRDSTOPTS = 0x17

	sysIPV6_CHECKSUM        = 0x18
	sysIPV6_RECVTCLASS      = 0x19
	sysIPV6_USE_MIN_MTU     = 0x20
	sysIPV6_DONTFRAG        = 0x21
	sysIPV6_SEC_OPT         = 0x22
	sysIPV6_SRC_PREFERENCES = 0x23
	sysIPV6_RECVPATHMTU     = 0x24
	sysIPV6_PATHMTU         = 0x25
	sysIPV6_TCLASS          = 0x26
	sysIPV6_V6ONLY          = 0x27

	sysIPV6_RECVDSTOPTS = 0x28

	sysMCAST_JOIN_GROUP         = 0x29
	sysMCAST_LEAVE_GROUP        = 0x2a
	sysMCAST_BLOCK_SOURCE       = 0x2b
	sysMCAST_UNBLOCK_SOURCE     = 0x2c
	sysMCAST_JOIN_SOURCE_GROUP  = 0x2d
	sysMCAST_LEAVE_SOURCE_GROUP = 0x2e

	sysIPV6_PREFER_SRC_HOME   = 0x1
	sysIPV6_PREFER_SRC_COA    = 0x2
	sysIPV6_PREFER_SRC_PUBLIC = 0x4
	sysIPV6_PREFER_SRC_TMP    = 0x8
	sysIPV6_PREFER_SRC_NONCGA = 0x10
	sysIPV6_PREFER_SRC_CGA    = 0x20

	sysIPV6_PREFER_SRC_MIPMASK    = 0x3
	sysIPV6_PREFER_SRC_MIPDEFAULT = 0x1
	sysIPV6_PREFER_SRC_TMPMASK    = 0xc
	sysIPV6_PREFER_SRC_TMPDEFAULT = 0x4
	sysIPV6_PREFER_SRC_CGAMASK    = 0x30
	sysIPV6_PREFER_SRC_CGADEFAULT = 0x10

	sysIPV6_PREFER_SRC_MASK = 0x3f

	sysIPV6_PREFER_SRC_DEFAULT = 0x15

	sysIPV6_BOUND_IF   = 0x41
	sysIPV6_UNSPEC_SRC = 0x42

	sysICMP6_FILTER = 0x1

	sizeofSockaddrStorage = 0x100
	sizeofSockaddrInet6   = 0x20
	sizeofInet6Pktinfo    = 0x14
	sizeofIPv6Mtuinfo     = 0x24

	sizeofIPv6Mreq       = 0x14
	sizeofGroupReq       = 0x104
	sizeofGroupSourceReq = 0x204

	sizeofICMPv6Filter = 0x20
)

type sockaddrStorage struct {
	Family     uint16
	X_ss_pad1  [6]int8
	X_ss_align float64
	X_ss_pad2  [240]int8
}

type sockaddrInet6 struct {
	Family         uint16
	Port           uint16
	Flowinfo       uint32
	Addr           [16]byte /* in6_addr */
	Scope_id       uint32
	X__sin6_src_id uint32
}

type inet6Pktinfo struct {
	Addr    [16]byte /* in6_addr */
	Ifindex uint32
}

type ipv6Mtuinfo struct {
	Addr sockaddrInet6
	Mtu  uint32
}

type ipv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Interface uint32
}

type groupReq struct {
	Interface uint32
	Pad_cgo_0 [256]byte
}

type groupSourceReq struct {
	Interface uint32
	Pad_cgo_0 [256]byte
	Pad_cgo_1 [256]byte
}

type icmpv6Filter struct {
	X__icmp6_filt [8]uint32
}
