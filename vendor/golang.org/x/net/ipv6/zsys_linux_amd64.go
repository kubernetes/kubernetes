// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_linux.go

package ipv6

const (
	sysIPV6_ADDRFORM       = 0x1
	sysIPV6_2292PKTINFO    = 0x2
	sysIPV6_2292HOPOPTS    = 0x3
	sysIPV6_2292DSTOPTS    = 0x4
	sysIPV6_2292RTHDR      = 0x5
	sysIPV6_2292PKTOPTIONS = 0x6
	sysIPV6_CHECKSUM       = 0x7
	sysIPV6_2292HOPLIMIT   = 0x8
	sysIPV6_NEXTHOP        = 0x9
	sysIPV6_FLOWINFO       = 0xb

	sysIPV6_UNICAST_HOPS        = 0x10
	sysIPV6_MULTICAST_IF        = 0x11
	sysIPV6_MULTICAST_HOPS      = 0x12
	sysIPV6_MULTICAST_LOOP      = 0x13
	sysIPV6_ADD_MEMBERSHIP      = 0x14
	sysIPV6_DROP_MEMBERSHIP     = 0x15
	sysMCAST_JOIN_GROUP         = 0x2a
	sysMCAST_LEAVE_GROUP        = 0x2d
	sysMCAST_JOIN_SOURCE_GROUP  = 0x2e
	sysMCAST_LEAVE_SOURCE_GROUP = 0x2f
	sysMCAST_BLOCK_SOURCE       = 0x2b
	sysMCAST_UNBLOCK_SOURCE     = 0x2c
	sysMCAST_MSFILTER           = 0x30
	sysIPV6_ROUTER_ALERT        = 0x16
	sysIPV6_MTU_DISCOVER        = 0x17
	sysIPV6_MTU                 = 0x18
	sysIPV6_RECVERR             = 0x19
	sysIPV6_V6ONLY              = 0x1a
	sysIPV6_JOIN_ANYCAST        = 0x1b
	sysIPV6_LEAVE_ANYCAST       = 0x1c

	sysIPV6_FLOWLABEL_MGR = 0x20
	sysIPV6_FLOWINFO_SEND = 0x21

	sysIPV6_IPSEC_POLICY = 0x22
	sysIPV6_XFRM_POLICY  = 0x23

	sysIPV6_RECVPKTINFO  = 0x31
	sysIPV6_PKTINFO      = 0x32
	sysIPV6_RECVHOPLIMIT = 0x33
	sysIPV6_HOPLIMIT     = 0x34
	sysIPV6_RECVHOPOPTS  = 0x35
	sysIPV6_HOPOPTS      = 0x36
	sysIPV6_RTHDRDSTOPTS = 0x37
	sysIPV6_RECVRTHDR    = 0x38
	sysIPV6_RTHDR        = 0x39
	sysIPV6_RECVDSTOPTS  = 0x3a
	sysIPV6_DSTOPTS      = 0x3b
	sysIPV6_RECVPATHMTU  = 0x3c
	sysIPV6_PATHMTU      = 0x3d
	sysIPV6_DONTFRAG     = 0x3e

	sysIPV6_RECVTCLASS = 0x42
	sysIPV6_TCLASS     = 0x43

	sysIPV6_ADDR_PREFERENCES = 0x48

	sysIPV6_PREFER_SRC_TMP            = 0x1
	sysIPV6_PREFER_SRC_PUBLIC         = 0x2
	sysIPV6_PREFER_SRC_PUBTMP_DEFAULT = 0x100
	sysIPV6_PREFER_SRC_COA            = 0x4
	sysIPV6_PREFER_SRC_HOME           = 0x400
	sysIPV6_PREFER_SRC_CGA            = 0x8
	sysIPV6_PREFER_SRC_NONCGA         = 0x800

	sysIPV6_MINHOPCOUNT = 0x49

	sysIPV6_ORIGDSTADDR     = 0x4a
	sysIPV6_RECVORIGDSTADDR = 0x4a
	sysIPV6_TRANSPARENT     = 0x4b
	sysIPV6_UNICAST_IF      = 0x4c

	sysICMPV6_FILTER = 0x1

	sysICMPV6_FILTER_BLOCK       = 0x1
	sysICMPV6_FILTER_PASS        = 0x2
	sysICMPV6_FILTER_BLOCKOTHERS = 0x3
	sysICMPV6_FILTER_PASSONLY    = 0x4

	sysSizeofKernelSockaddrStorage = 0x80
	sysSizeofSockaddrInet6         = 0x1c
	sysSizeofInet6Pktinfo          = 0x14
	sysSizeofIPv6Mtuinfo           = 0x20
	sysSizeofIPv6FlowlabelReq      = 0x20

	sysSizeofIPv6Mreq       = 0x14
	sysSizeofGroupReq       = 0x88
	sysSizeofGroupSourceReq = 0x108

	sysSizeofICMPv6Filter = 0x20
)

type sysKernelSockaddrStorage struct {
	Family  uint16
	X__data [126]int8
}

type sysSockaddrInet6 struct {
	Family   uint16
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte /* in6_addr */
	Scope_id uint32
}

type sysInet6Pktinfo struct {
	Addr    [16]byte /* in6_addr */
	Ifindex int32
}

type sysIPv6Mtuinfo struct {
	Addr sysSockaddrInet6
	Mtu  uint32
}

type sysIPv6FlowlabelReq struct {
	Dst        [16]byte /* in6_addr */
	Label      uint32
	Action     uint8
	Share      uint8
	Flags      uint16
	Expires    uint16
	Linger     uint16
	X__flr_pad uint32
}

type sysIPv6Mreq struct {
	Multiaddr [16]byte /* in6_addr */
	Ifindex   int32
}

type sysGroupReq struct {
	Interface uint32
	Pad_cgo_0 [4]byte
	Group     sysKernelSockaddrStorage
}

type sysGroupSourceReq struct {
	Interface uint32
	Pad_cgo_0 [4]byte
	Group     sysKernelSockaddrStorage
	Source    sysKernelSockaddrStorage
}

type sysICMPv6Filter struct {
	Data [8]uint32
}
