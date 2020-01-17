// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// +godefs map struct_in_addr [4]byte /* in_addr */

package ipv4

/*
#include <sys/socket.h>

#include <netinet/in.h>
*/
import "C"

const (
	sysIP_OPTIONS     = C.IP_OPTIONS
	sysIP_HDRINCL     = C.IP_HDRINCL
	sysIP_TOS         = C.IP_TOS
	sysIP_TTL         = C.IP_TTL
	sysIP_RECVOPTS    = C.IP_RECVOPTS
	sysIP_RECVRETOPTS = C.IP_RECVRETOPTS
	sysIP_RECVDSTADDR = C.IP_RECVDSTADDR
	sysIP_RETOPTS     = C.IP_RETOPTS
	sysIP_RECVIF      = C.IP_RECVIF
	sysIP_STRIPHDR    = C.IP_STRIPHDR
	sysIP_RECVTTL     = C.IP_RECVTTL
	sysIP_BOUND_IF    = C.IP_BOUND_IF
	sysIP_PKTINFO     = C.IP_PKTINFO
	sysIP_RECVPKTINFO = C.IP_RECVPKTINFO

	sysIP_MULTICAST_IF           = C.IP_MULTICAST_IF
	sysIP_MULTICAST_TTL          = C.IP_MULTICAST_TTL
	sysIP_MULTICAST_LOOP         = C.IP_MULTICAST_LOOP
	sysIP_ADD_MEMBERSHIP         = C.IP_ADD_MEMBERSHIP
	sysIP_DROP_MEMBERSHIP        = C.IP_DROP_MEMBERSHIP
	sysIP_MULTICAST_VIF          = C.IP_MULTICAST_VIF
	sysIP_MULTICAST_IFINDEX      = C.IP_MULTICAST_IFINDEX
	sysIP_ADD_SOURCE_MEMBERSHIP  = C.IP_ADD_SOURCE_MEMBERSHIP
	sysIP_DROP_SOURCE_MEMBERSHIP = C.IP_DROP_SOURCE_MEMBERSHIP
	sysIP_BLOCK_SOURCE           = C.IP_BLOCK_SOURCE
	sysIP_UNBLOCK_SOURCE         = C.IP_UNBLOCK_SOURCE
	sysMCAST_JOIN_GROUP          = C.MCAST_JOIN_GROUP
	sysMCAST_LEAVE_GROUP         = C.MCAST_LEAVE_GROUP
	sysMCAST_JOIN_SOURCE_GROUP   = C.MCAST_JOIN_SOURCE_GROUP
	sysMCAST_LEAVE_SOURCE_GROUP  = C.MCAST_LEAVE_SOURCE_GROUP
	sysMCAST_BLOCK_SOURCE        = C.MCAST_BLOCK_SOURCE
	sysMCAST_UNBLOCK_SOURCE      = C.MCAST_UNBLOCK_SOURCE

	sizeofSockaddrStorage = C.sizeof_struct_sockaddr_storage
	sizeofSockaddrInet    = C.sizeof_struct_sockaddr_in
	sizeofInetPktinfo     = C.sizeof_struct_in_pktinfo

	sizeofIPMreq         = C.sizeof_struct_ip_mreq
	sizeofIPMreqn        = C.sizeof_struct_ip_mreqn
	sizeofIPMreqSource   = C.sizeof_struct_ip_mreq_source
	sizeofGroupReq       = C.sizeof_struct_group_req
	sizeofGroupSourceReq = C.sizeof_struct_group_source_req
)

type sockaddrStorage C.struct_sockaddr_storage

type sockaddrInet C.struct_sockaddr_in

type inetPktinfo C.struct_in_pktinfo

type ipMreq C.struct_ip_mreq

type ipMreqn C.struct_ip_mreqn

type ipMreqSource C.struct_ip_mreq_source

type groupReq C.struct_group_req

type groupSourceReq C.struct_group_source_req
