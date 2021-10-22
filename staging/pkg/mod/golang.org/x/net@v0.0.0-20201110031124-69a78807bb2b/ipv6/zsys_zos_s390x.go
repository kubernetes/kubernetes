// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hand edited based on zerrors_zos_s390x.go
// TODO(Bill O'Farrell): auto-generate.

package ipv6

const (
	sysIPV6_ADDR_PREFERENCES  = 32
	sysIPV6_CHECKSUM          = 19
	sysIPV6_DONTFRAG          = 29
	sysIPV6_DSTOPTS           = 23
	sysIPV6_HOPLIMIT          = 11
	sysIPV6_HOPOPTS           = 22
	sysIPV6_JOIN_GROUP        = 5
	sysIPV6_LEAVE_GROUP       = 6
	sysIPV6_MULTICAST_HOPS    = 9
	sysIPV6_MULTICAST_IF      = 7
	sysIPV6_MULTICAST_LOOP    = 4
	sysIPV6_NEXTHOP           = 20
	sysIPV6_PATHMTU           = 12
	sysIPV6_PKTINFO           = 13
	sysIPV6_PREFER_SRC_CGA    = 0x10
	sysIPV6_PREFER_SRC_COA    = 0x02
	sysIPV6_PREFER_SRC_HOME   = 0x01
	sysIPV6_PREFER_SRC_NONCGA = 0x20
	sysIPV6_PREFER_SRC_PUBLIC = 0x08
	sysIPV6_PREFER_SRC_TMP    = 0x04
	sysIPV6_RECVDSTOPTS       = 28
	sysIPV6_RECVHOPLIMIT      = 14
	sysIPV6_RECVHOPOPTS       = 26
	sysIPV6_RECVPATHMTU       = 16
	sysIPV6_RECVPKTINFO       = 15
	sysIPV6_RECVRTHDR         = 25
	sysIPV6_RECVTCLASS        = 31
	sysIPV6_RTHDR             = 21
	sysIPV6_RTHDRDSTOPTS      = 24
	sysIPV6_RTHDR_TYPE_0      = 0
	sysIPV6_TCLASS            = 30
	sysIPV6_UNICAST_HOPS      = 3
	sysIPV6_USE_MIN_MTU       = 18
	sysIPV6_V6ONLY            = 10

	sysMCAST_JOIN_GROUP         = 40
	sysMCAST_LEAVE_GROUP        = 41
	sysMCAST_JOIN_SOURCE_GROUP  = 42
	sysMCAST_LEAVE_SOURCE_GROUP = 43
	sysMCAST_BLOCK_SOURCE       = 44
	sysMCAST_UNBLOCK_SOURCE     = 45

	sysICMP6_FILTER = 0x1

	sizeofSockaddrStorage = 128
	sizeofICMPv6Filter    = 32
	sizeofInet6Pktinfo    = 20
	sizeofIPv6Mtuinfo     = 32
	sizeofSockaddrInet6   = 28
	sizeofGroupReq        = 136
	sizeofGroupSourceReq  = 264
)

type sockaddrStorage struct {
	Len      uint8
	Family   byte
	ss_pad1  [6]byte
	ss_align int64
	ss_pad2  [112]byte
}

type sockaddrInet6 struct {
	Len      uint8
	Family   uint8
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte
	Scope_id uint32
}

type inet6Pktinfo struct {
	Addr    [16]byte
	Ifindex uint32
}

type ipv6Mtuinfo struct {
	Addr sockaddrInet6
	Mtu  uint32
}

type groupReq struct {
	Interface uint32
	reserved  uint32
	Group     sockaddrStorage
}

type groupSourceReq struct {
	Interface uint32
	reserved  uint32
	Group     sockaddrStorage
	Source    sockaddrStorage
}

type icmpv6Filter struct {
	Filt [8]uint32
}
