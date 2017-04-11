// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// +godefs map struct_in6_addr [16]byte /* in6_addr */

package ipv6

/*
#include <sys/param.h>
#include <sys/socket.h>

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
	sysIPV6_PORTRANGE      = C.IPV6_PORTRANGE
	sysICMP6_FILTER        = C.ICMP6_FILTER

	sysIPV6_CHECKSUM = C.IPV6_CHECKSUM
	sysIPV6_V6ONLY   = C.IPV6_V6ONLY

	sysIPV6_IPSEC_POLICY = C.IPV6_IPSEC_POLICY

	sysIPV6_RTHDRDSTOPTS = C.IPV6_RTHDRDSTOPTS

	sysIPV6_RECVPKTINFO  = C.IPV6_RECVPKTINFO
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

	sysIPV6_RECVTCLASS = C.IPV6_RECVTCLASS

	sysIPV6_AUTOFLOWLABEL = C.IPV6_AUTOFLOWLABEL

	sysIPV6_TCLASS   = C.IPV6_TCLASS
	sysIPV6_DONTFRAG = C.IPV6_DONTFRAG

	sysIPV6_PREFER_TEMPADDR = C.IPV6_PREFER_TEMPADDR

	sysIPV6_BINDANY = C.IPV6_BINDANY

	sysIPV6_MSFILTER = C.IPV6_MSFILTER

	sysMCAST_JOIN_GROUP         = C.MCAST_JOIN_GROUP
	sysMCAST_LEAVE_GROUP        = C.MCAST_LEAVE_GROUP
	sysMCAST_JOIN_SOURCE_GROUP  = C.MCAST_JOIN_SOURCE_GROUP
	sysMCAST_LEAVE_SOURCE_GROUP = C.MCAST_LEAVE_SOURCE_GROUP
	sysMCAST_BLOCK_SOURCE       = C.MCAST_BLOCK_SOURCE
	sysMCAST_UNBLOCK_SOURCE     = C.MCAST_UNBLOCK_SOURCE

	sysIPV6_PORTRANGE_DEFAULT = C.IPV6_PORTRANGE_DEFAULT
	sysIPV6_PORTRANGE_HIGH    = C.IPV6_PORTRANGE_HIGH
	sysIPV6_PORTRANGE_LOW     = C.IPV6_PORTRANGE_LOW

	sysSizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sysSizeofSockaddrInet6   = C.sizeof_struct_sockaddr_in6
	sysSizeofInet6Pktinfo    = C.sizeof_struct_in6_pktinfo
	sysSizeofIPv6Mtuinfo     = C.sizeof_struct_ip6_mtuinfo

	sysSizeofIPv6Mreq       = C.sizeof_struct_ipv6_mreq
	sysSizeofGroupReq       = C.sizeof_struct_group_req
	sysSizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sysSizeofICMPv6Filter = C.sizeof_struct_icmp6_filter
)

type sysSockaddrStorage C.struct_sockaddr_storage

type sysSockaddrInet6 C.struct_sockaddr_in6

type sysInet6Pktinfo C.struct_in6_pktinfo

type sysIPv6Mtuinfo C.struct_ip6_mtuinfo

type sysIPv6Mreq C.struct_ipv6_mreq

type sysGroupReq C.struct_group_req

type sysGroupSourceReq C.struct_group_source_req

type sysICMPv6Filter C.struct_icmp6_filter
