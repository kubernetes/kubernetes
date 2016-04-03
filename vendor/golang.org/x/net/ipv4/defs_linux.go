// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// +godefs map struct_in_addr [4]byte /* in_addr */

package ipv4

/*
#include <time.h>

#include <linux/errqueue.h>
#include <linux/icmp.h>
#include <linux/in.h>
*/
import "C"

const (
	sysIP_TOS             = C.IP_TOS
	sysIP_TTL             = C.IP_TTL
	sysIP_HDRINCL         = C.IP_HDRINCL
	sysIP_OPTIONS         = C.IP_OPTIONS
	sysIP_ROUTER_ALERT    = C.IP_ROUTER_ALERT
	sysIP_RECVOPTS        = C.IP_RECVOPTS
	sysIP_RETOPTS         = C.IP_RETOPTS
	sysIP_PKTINFO         = C.IP_PKTINFO
	sysIP_PKTOPTIONS      = C.IP_PKTOPTIONS
	sysIP_MTU_DISCOVER    = C.IP_MTU_DISCOVER
	sysIP_RECVERR         = C.IP_RECVERR
	sysIP_RECVTTL         = C.IP_RECVTTL
	sysIP_RECVTOS         = C.IP_RECVTOS
	sysIP_MTU             = C.IP_MTU
	sysIP_FREEBIND        = C.IP_FREEBIND
	sysIP_TRANSPARENT     = C.IP_TRANSPARENT
	sysIP_RECVRETOPTS     = C.IP_RECVRETOPTS
	sysIP_ORIGDSTADDR     = C.IP_ORIGDSTADDR
	sysIP_RECVORIGDSTADDR = C.IP_RECVORIGDSTADDR
	sysIP_MINTTL          = C.IP_MINTTL
	sysIP_NODEFRAG        = C.IP_NODEFRAG
	sysIP_UNICAST_IF      = C.IP_UNICAST_IF

	sysIP_MULTICAST_IF           = C.IP_MULTICAST_IF
	sysIP_MULTICAST_TTL          = C.IP_MULTICAST_TTL
	sysIP_MULTICAST_LOOP         = C.IP_MULTICAST_LOOP
	sysIP_ADD_MEMBERSHIP         = C.IP_ADD_MEMBERSHIP
	sysIP_DROP_MEMBERSHIP        = C.IP_DROP_MEMBERSHIP
	sysIP_UNBLOCK_SOURCE         = C.IP_UNBLOCK_SOURCE
	sysIP_BLOCK_SOURCE           = C.IP_BLOCK_SOURCE
	sysIP_ADD_SOURCE_MEMBERSHIP  = C.IP_ADD_SOURCE_MEMBERSHIP
	sysIP_DROP_SOURCE_MEMBERSHIP = C.IP_DROP_SOURCE_MEMBERSHIP
	sysIP_MSFILTER               = C.IP_MSFILTER
	sysMCAST_JOIN_GROUP          = C.MCAST_JOIN_GROUP
	sysMCAST_LEAVE_GROUP         = C.MCAST_LEAVE_GROUP
	sysMCAST_JOIN_SOURCE_GROUP   = C.MCAST_JOIN_SOURCE_GROUP
	sysMCAST_LEAVE_SOURCE_GROUP  = C.MCAST_LEAVE_SOURCE_GROUP
	sysMCAST_BLOCK_SOURCE        = C.MCAST_BLOCK_SOURCE
	sysMCAST_UNBLOCK_SOURCE      = C.MCAST_UNBLOCK_SOURCE
	sysMCAST_MSFILTER            = C.MCAST_MSFILTER
	sysIP_MULTICAST_ALL          = C.IP_MULTICAST_ALL

	//sysIP_PMTUDISC_DONT      = C.IP_PMTUDISC_DONT
	//sysIP_PMTUDISC_WANT      = C.IP_PMTUDISC_WANT
	//sysIP_PMTUDISC_DO        = C.IP_PMTUDISC_DO
	//sysIP_PMTUDISC_PROBE     = C.IP_PMTUDISC_PROBE
	//sysIP_PMTUDISC_INTERFACE = C.IP_PMTUDISC_INTERFACE
	//sysIP_PMTUDISC_OMIT      = C.IP_PMTUDISC_OMIT

	sysICMP_FILTER = C.ICMP_FILTER

	sysSO_EE_ORIGIN_NONE         = C.SO_EE_ORIGIN_NONE
	sysSO_EE_ORIGIN_LOCAL        = C.SO_EE_ORIGIN_LOCAL
	sysSO_EE_ORIGIN_ICMP         = C.SO_EE_ORIGIN_ICMP
	sysSO_EE_ORIGIN_ICMP6        = C.SO_EE_ORIGIN_ICMP6
	sysSO_EE_ORIGIN_TXSTATUS     = C.SO_EE_ORIGIN_TXSTATUS
	sysSO_EE_ORIGIN_TIMESTAMPING = C.SO_EE_ORIGIN_TIMESTAMPING

	sysSizeofKernelSockaddrStorage = C.sizeof_struct___kernel_sockaddr_storage
	sysSizeofSockaddrInet          = C.sizeof_struct_sockaddr_in
	sysSizeofInetPktinfo           = C.sizeof_struct_in_pktinfo
	sysSizeofSockExtendedErr       = C.sizeof_struct_sock_extended_err

	sysSizeofIPMreq         = C.sizeof_struct_ip_mreq
	sysSizeofIPMreqn        = C.sizeof_struct_ip_mreqn
	sysSizeofIPMreqSource   = C.sizeof_struct_ip_mreq_source
	sysSizeofGroupReq       = C.sizeof_struct_group_req
	sysSizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sysSizeofICMPFilter = C.sizeof_struct_icmp_filter
)

type sysKernelSockaddrStorage C.struct___kernel_sockaddr_storage

type sysSockaddrInet C.struct_sockaddr_in

type sysInetPktinfo C.struct_in_pktinfo

type sysSockExtendedErr C.struct_sock_extended_err

type sysIPMreq C.struct_ip_mreq

type sysIPMreqn C.struct_ip_mreqn

type sysIPMreqSource C.struct_ip_mreq_source

type sysGroupReq C.struct_group_req

type sysGroupSourceReq C.struct_group_source_req

type sysICMPFilter C.struct_icmp_filter
