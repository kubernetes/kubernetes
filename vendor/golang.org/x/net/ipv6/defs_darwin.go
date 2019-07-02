// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// +godefs map struct_in6_addr [16]byte /* in6_addr */

package ipv6

/*
#define __APPLE_USE_RFC_3542
#include <netinet/in.h>
#include <netinet/icmp6.h>
*/
import "C"

const (
	sysIPV6_UNICAST_HOPS   = C.IPV6_UNICAST_HOPS
	sysIPV6_MULTICAST_IF   = C.IPV6_MULTICAST_IF
	sysIPV6_MULTICAST_HOPS = C.IPV6_MULTICAST_HOPS
	sysIPV6_MULTICAST_LOOP = C.IPV6_MULTICAST_LOOP
	sysIPV6_JOIN_GROUP     = C.IPV6_JOIN_GROUP
	sysIPV6_LEAVE_GROUP    = C.IPV6_LEAVE_GROUP

	sysIPV6_PORTRANGE    = C.IPV6_PORTRANGE
	sysICMP6_FILTER      = C.ICMP6_FILTER
	sysIPV6_2292PKTINFO  = C.IPV6_2292PKTINFO
	sysIPV6_2292HOPLIMIT = C.IPV6_2292HOPLIMIT
	sysIPV6_2292NEXTHOP  = C.IPV6_2292NEXTHOP
	sysIPV6_2292HOPOPTS  = C.IPV6_2292HOPOPTS
	sysIPV6_2292DSTOPTS  = C.IPV6_2292DSTOPTS
	sysIPV6_2292RTHDR    = C.IPV6_2292RTHDR

	sysIPV6_2292PKTOPTIONS = C.IPV6_2292PKTOPTIONS

	sysIPV6_CHECKSUM = C.IPV6_CHECKSUM
	sysIPV6_V6ONLY   = C.IPV6_V6ONLY

	sysIPV6_IPSEC_POLICY = C.IPV6_IPSEC_POLICY

	sysIPV6_RECVTCLASS = C.IPV6_RECVTCLASS
	sysIPV6_TCLASS     = C.IPV6_TCLASS

	sysIPV6_RTHDRDSTOPTS = C.IPV6_RTHDRDSTOPTS

	sysIPV6_RECVPKTINFO = C.IPV6_RECVPKTINFO

	sysIPV6_RECVHOPLIMIT = C.IPV6_RECVHOPLIMIT
	sysIPV6_RECVRTHDR    = C.IPV6_RECVRTHDR
	sysIPV6_RECVHOPOPTS  = C.IPV6_RECVHOPOPTS
	sysIPV6_RECVDSTOPTS  = C.IPV6_RECVDSTOPTS

	sysIPV6_USE_MIN_MTU = C.IPV6_USE_MIN_MTU
	sysIPV6_RECVPATHMTU = C.IPV6_RECVPATHMTU

	sysIPV6_PATHMTU = C.IPV6_PATHMTU

	sysIPV6_PKTINFO  = C.IPV6_PKTINFO
	sysIPV6_HOPLIMIT = C.IPV6_HOPLIMIT
	sysIPV6_NEXTHOP  = C.IPV6_NEXTHOP
	sysIPV6_HOPOPTS  = C.IPV6_HOPOPTS
	sysIPV6_DSTOPTS  = C.IPV6_DSTOPTS
	sysIPV6_RTHDR    = C.IPV6_RTHDR

	sysIPV6_AUTOFLOWLABEL = C.IPV6_AUTOFLOWLABEL

	sysIPV6_DONTFRAG = C.IPV6_DONTFRAG

	sysIPV6_PREFER_TEMPADDR = C.IPV6_PREFER_TEMPADDR

	sysIPV6_MSFILTER            = C.IPV6_MSFILTER
	sysMCAST_JOIN_GROUP         = C.MCAST_JOIN_GROUP
	sysMCAST_LEAVE_GROUP        = C.MCAST_LEAVE_GROUP
	sysMCAST_JOIN_SOURCE_GROUP  = C.MCAST_JOIN_SOURCE_GROUP
	sysMCAST_LEAVE_SOURCE_GROUP = C.MCAST_LEAVE_SOURCE_GROUP
	sysMCAST_BLOCK_SOURCE       = C.MCAST_BLOCK_SOURCE
	sysMCAST_UNBLOCK_SOURCE     = C.MCAST_UNBLOCK_SOURCE

	sysIPV6_BOUND_IF = C.IPV6_BOUND_IF

	sysIPV6_PORTRANGE_DEFAULT = C.IPV6_PORTRANGE_DEFAULT
	sysIPV6_PORTRANGE_HIGH    = C.IPV6_PORTRANGE_HIGH
	sysIPV6_PORTRANGE_LOW     = C.IPV6_PORTRANGE_LOW

	sizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sizeofSockaddrInet6   = C.sizeof_struct_sockaddr_in6
	sizeofInet6Pktinfo    = C.sizeof_struct_in6_pktinfo
	sizeofIPv6Mtuinfo     = C.sizeof_struct_ip6_mtuinfo

	sizeofIPv6Mreq       = C.sizeof_struct_ipv6_mreq
	sizeofGroupReq       = C.sizeof_struct_group_req
	sizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sizeofICMPv6Filter = C.sizeof_struct_icmp6_filter
)

type sockaddrStorage C.struct_sockaddr_storage

type sockaddrInet6 C.struct_sockaddr_in6

type inet6Pktinfo C.struct_in6_pktinfo

type ipv6Mtuinfo C.struct_ip6_mtuinfo

type ipv6Mreq C.struct_ipv6_mreq

type icmpv6Filter C.struct_icmp6_filter

type groupReq C.struct_group_req

type groupSourceReq C.struct_group_source_req
