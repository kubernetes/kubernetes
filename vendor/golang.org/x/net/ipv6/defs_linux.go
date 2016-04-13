// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// +godefs map struct_in6_addr [16]byte /* in6_addr */

package ipv6

/*
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/ipv6.h>
#include <linux/icmpv6.h>
*/
import "C"

const (
	sysIPV6_ADDRFORM       = C.IPV6_ADDRFORM
	sysIPV6_2292PKTINFO    = C.IPV6_2292PKTINFO
	sysIPV6_2292HOPOPTS    = C.IPV6_2292HOPOPTS
	sysIPV6_2292DSTOPTS    = C.IPV6_2292DSTOPTS
	sysIPV6_2292RTHDR      = C.IPV6_2292RTHDR
	sysIPV6_2292PKTOPTIONS = C.IPV6_2292PKTOPTIONS
	sysIPV6_CHECKSUM       = C.IPV6_CHECKSUM
	sysIPV6_2292HOPLIMIT   = C.IPV6_2292HOPLIMIT
	sysIPV6_NEXTHOP        = C.IPV6_NEXTHOP
	sysIPV6_FLOWINFO       = C.IPV6_FLOWINFO

	sysIPV6_UNICAST_HOPS        = C.IPV6_UNICAST_HOPS
	sysIPV6_MULTICAST_IF        = C.IPV6_MULTICAST_IF
	sysIPV6_MULTICAST_HOPS      = C.IPV6_MULTICAST_HOPS
	sysIPV6_MULTICAST_LOOP      = C.IPV6_MULTICAST_LOOP
	sysIPV6_ADD_MEMBERSHIP      = C.IPV6_ADD_MEMBERSHIP
	sysIPV6_DROP_MEMBERSHIP     = C.IPV6_DROP_MEMBERSHIP
	sysMCAST_JOIN_GROUP         = C.MCAST_JOIN_GROUP
	sysMCAST_LEAVE_GROUP        = C.MCAST_LEAVE_GROUP
	sysMCAST_JOIN_SOURCE_GROUP  = C.MCAST_JOIN_SOURCE_GROUP
	sysMCAST_LEAVE_SOURCE_GROUP = C.MCAST_LEAVE_SOURCE_GROUP
	sysMCAST_BLOCK_SOURCE       = C.MCAST_BLOCK_SOURCE
	sysMCAST_UNBLOCK_SOURCE     = C.MCAST_UNBLOCK_SOURCE
	sysMCAST_MSFILTER           = C.MCAST_MSFILTER
	sysIPV6_ROUTER_ALERT        = C.IPV6_ROUTER_ALERT
	sysIPV6_MTU_DISCOVER        = C.IPV6_MTU_DISCOVER
	sysIPV6_MTU                 = C.IPV6_MTU
	sysIPV6_RECVERR             = C.IPV6_RECVERR
	sysIPV6_V6ONLY              = C.IPV6_V6ONLY
	sysIPV6_JOIN_ANYCAST        = C.IPV6_JOIN_ANYCAST
	sysIPV6_LEAVE_ANYCAST       = C.IPV6_LEAVE_ANYCAST

	//sysIPV6_PMTUDISC_DONT      = C.IPV6_PMTUDISC_DONT
	//sysIPV6_PMTUDISC_WANT      = C.IPV6_PMTUDISC_WANT
	//sysIPV6_PMTUDISC_DO        = C.IPV6_PMTUDISC_DO
	//sysIPV6_PMTUDISC_PROBE     = C.IPV6_PMTUDISC_PROBE
	//sysIPV6_PMTUDISC_INTERFACE = C.IPV6_PMTUDISC_INTERFACE
	//sysIPV6_PMTUDISC_OMIT      = C.IPV6_PMTUDISC_OMIT

	sysIPV6_FLOWLABEL_MGR = C.IPV6_FLOWLABEL_MGR
	sysIPV6_FLOWINFO_SEND = C.IPV6_FLOWINFO_SEND

	sysIPV6_IPSEC_POLICY = C.IPV6_IPSEC_POLICY
	sysIPV6_XFRM_POLICY  = C.IPV6_XFRM_POLICY

	sysIPV6_RECVPKTINFO  = C.IPV6_RECVPKTINFO
	sysIPV6_PKTINFO      = C.IPV6_PKTINFO
	sysIPV6_RECVHOPLIMIT = C.IPV6_RECVHOPLIMIT
	sysIPV6_HOPLIMIT     = C.IPV6_HOPLIMIT
	sysIPV6_RECVHOPOPTS  = C.IPV6_RECVHOPOPTS
	sysIPV6_HOPOPTS      = C.IPV6_HOPOPTS
	sysIPV6_RTHDRDSTOPTS = C.IPV6_RTHDRDSTOPTS
	sysIPV6_RECVRTHDR    = C.IPV6_RECVRTHDR
	sysIPV6_RTHDR        = C.IPV6_RTHDR
	sysIPV6_RECVDSTOPTS  = C.IPV6_RECVDSTOPTS
	sysIPV6_DSTOPTS      = C.IPV6_DSTOPTS
	sysIPV6_RECVPATHMTU  = C.IPV6_RECVPATHMTU
	sysIPV6_PATHMTU      = C.IPV6_PATHMTU
	sysIPV6_DONTFRAG     = C.IPV6_DONTFRAG

	sysIPV6_RECVTCLASS = C.IPV6_RECVTCLASS
	sysIPV6_TCLASS     = C.IPV6_TCLASS

	sysIPV6_ADDR_PREFERENCES = C.IPV6_ADDR_PREFERENCES

	sysIPV6_PREFER_SRC_TMP            = C.IPV6_PREFER_SRC_TMP
	sysIPV6_PREFER_SRC_PUBLIC         = C.IPV6_PREFER_SRC_PUBLIC
	sysIPV6_PREFER_SRC_PUBTMP_DEFAULT = C.IPV6_PREFER_SRC_PUBTMP_DEFAULT
	sysIPV6_PREFER_SRC_COA            = C.IPV6_PREFER_SRC_COA
	sysIPV6_PREFER_SRC_HOME           = C.IPV6_PREFER_SRC_HOME
	sysIPV6_PREFER_SRC_CGA            = C.IPV6_PREFER_SRC_CGA
	sysIPV6_PREFER_SRC_NONCGA         = C.IPV6_PREFER_SRC_NONCGA

	sysIPV6_MINHOPCOUNT = C.IPV6_MINHOPCOUNT

	sysIPV6_ORIGDSTADDR     = C.IPV6_ORIGDSTADDR
	sysIPV6_RECVORIGDSTADDR = C.IPV6_RECVORIGDSTADDR
	sysIPV6_TRANSPARENT     = C.IPV6_TRANSPARENT
	sysIPV6_UNICAST_IF      = C.IPV6_UNICAST_IF

	sysICMPV6_FILTER = C.ICMPV6_FILTER

	sysICMPV6_FILTER_BLOCK       = C.ICMPV6_FILTER_BLOCK
	sysICMPV6_FILTER_PASS        = C.ICMPV6_FILTER_PASS
	sysICMPV6_FILTER_BLOCKOTHERS = C.ICMPV6_FILTER_BLOCKOTHERS
	sysICMPV6_FILTER_PASSONLY    = C.ICMPV6_FILTER_PASSONLY

	sysSizeofKernelSockaddrStorage = C.sizeof_struct___kernel_sockaddr_storage
	sysSizeofSockaddrInet6         = C.sizeof_struct_sockaddr_in6
	sysSizeofInet6Pktinfo          = C.sizeof_struct_in6_pktinfo
	sysSizeofIPv6Mtuinfo           = C.sizeof_struct_ip6_mtuinfo
	sysSizeofIPv6FlowlabelReq      = C.sizeof_struct_in6_flowlabel_req

	sysSizeofIPv6Mreq       = C.sizeof_struct_ipv6_mreq
	sysSizeofGroupReq       = C.sizeof_struct_group_req
	sysSizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sysSizeofICMPv6Filter = C.sizeof_struct_icmp6_filter
)

type sysKernelSockaddrStorage C.struct___kernel_sockaddr_storage

type sysSockaddrInet6 C.struct_sockaddr_in6

type sysInet6Pktinfo C.struct_in6_pktinfo

type sysIPv6Mtuinfo C.struct_ip6_mtuinfo

type sysIPv6FlowlabelReq C.struct_in6_flowlabel_req

type sysIPv6Mreq C.struct_ipv6_mreq

type sysGroupReq C.struct_group_req

type sysGroupSourceReq C.struct_group_source_req

type sysICMPv6Filter C.struct_icmp6_filter
