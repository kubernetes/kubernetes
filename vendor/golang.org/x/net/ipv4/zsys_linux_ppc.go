// Created by cgo -godefs - DO NOT EDIT
// cgo -godefs defs_linux.go

package ipv4

const (
	sysIP_TOS             = 0x1
	sysIP_TTL             = 0x2
	sysIP_HDRINCL         = 0x3
	sysIP_OPTIONS         = 0x4
	sysIP_ROUTER_ALERT    = 0x5
	sysIP_RECVOPTS        = 0x6
	sysIP_RETOPTS         = 0x7
	sysIP_PKTINFO         = 0x8
	sysIP_PKTOPTIONS      = 0x9
	sysIP_MTU_DISCOVER    = 0xa
	sysIP_RECVERR         = 0xb
	sysIP_RECVTTL         = 0xc
	sysIP_RECVTOS         = 0xd
	sysIP_MTU             = 0xe
	sysIP_FREEBIND        = 0xf
	sysIP_TRANSPARENT     = 0x13
	sysIP_RECVRETOPTS     = 0x7
	sysIP_ORIGDSTADDR     = 0x14
	sysIP_RECVORIGDSTADDR = 0x14
	sysIP_MINTTL          = 0x15
	sysIP_NODEFRAG        = 0x16
	sysIP_UNICAST_IF      = 0x32

	sysIP_MULTICAST_IF           = 0x20
	sysIP_MULTICAST_TTL          = 0x21
	sysIP_MULTICAST_LOOP         = 0x22
	sysIP_ADD_MEMBERSHIP         = 0x23
	sysIP_DROP_MEMBERSHIP        = 0x24
	sysIP_UNBLOCK_SOURCE         = 0x25
	sysIP_BLOCK_SOURCE           = 0x26
	sysIP_ADD_SOURCE_MEMBERSHIP  = 0x27
	sysIP_DROP_SOURCE_MEMBERSHIP = 0x28
	sysIP_MSFILTER               = 0x29
	sysMCAST_JOIN_GROUP          = 0x2a
	sysMCAST_LEAVE_GROUP         = 0x2d
	sysMCAST_JOIN_SOURCE_GROUP   = 0x2e
	sysMCAST_LEAVE_SOURCE_GROUP  = 0x2f
	sysMCAST_BLOCK_SOURCE        = 0x2b
	sysMCAST_UNBLOCK_SOURCE      = 0x2c
	sysMCAST_MSFILTER            = 0x30
	sysIP_MULTICAST_ALL          = 0x31

	sysICMP_FILTER = 0x1

	sysSO_EE_ORIGIN_NONE         = 0x0
	sysSO_EE_ORIGIN_LOCAL        = 0x1
	sysSO_EE_ORIGIN_ICMP         = 0x2
	sysSO_EE_ORIGIN_ICMP6        = 0x3
	sysSO_EE_ORIGIN_TXSTATUS     = 0x4
	sysSO_EE_ORIGIN_TIMESTAMPING = 0x4

	sysSOL_SOCKET       = 0x1
	sysSO_ATTACH_FILTER = 0x1a

	sizeofKernelSockaddrStorage = 0x80
	sizeofSockaddrInet          = 0x10
	sizeofInetPktinfo           = 0xc
	sizeofSockExtendedErr       = 0x10

	sizeofIPMreq         = 0x8
	sizeofIPMreqn        = 0xc
	sizeofIPMreqSource   = 0xc
	sizeofGroupReq       = 0x84
	sizeofGroupSourceReq = 0x104

	sizeofICMPFilter = 0x4

	sizeofSockFprog = 0x8
)

type kernelSockaddrStorage struct {
	Family  uint16
	X__data [126]uint8
}

type sockaddrInet struct {
	Family uint16
	Port   uint16
	Addr   [4]byte /* in_addr */
	X__pad [8]uint8
}

type inetPktinfo struct {
	Ifindex  int32
	Spec_dst [4]byte /* in_addr */
	Addr     [4]byte /* in_addr */
}

type sockExtendedErr struct {
	Errno  uint32
	Origin uint8
	Type   uint8
	Code   uint8
	Pad    uint8
	Info   uint32
	Data   uint32
}

type ipMreq struct {
	Multiaddr [4]byte /* in_addr */
	Interface [4]byte /* in_addr */
}

type ipMreqn struct {
	Multiaddr [4]byte /* in_addr */
	Address   [4]byte /* in_addr */
	Ifindex   int32
}

type ipMreqSource struct {
	Multiaddr  uint32
	Interface  uint32
	Sourceaddr uint32
}

type groupReq struct {
	Interface uint32
	Group     kernelSockaddrStorage
}

type groupSourceReq struct {
	Interface uint32
	Group     kernelSockaddrStorage
	Source    kernelSockaddrStorage
}

type icmpFilter struct {
	Data uint32
}

type sockFProg struct {
	Len       uint16
	Pad_cgo_0 [2]byte
	Filter    *sockFilter
}

type sockFilter struct {
	Code uint16
	Jt   uint8
	Jf   uint8
	K    uint32
}
