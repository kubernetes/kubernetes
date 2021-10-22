// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// +godefs map struct_in6_addr [16]byte /* in6_addr */

package ipv6

/*
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/ipv6.h>
#include <linux/icmpv6.h>
#include <linux/filter.h>
#include <sys/socket.h>
*/
import "C"

const (
	sizeofKernelSockaddrStorage = C.sizeof_struct___kernel_sockaddr_storage
	sizeofSockaddrInet6         = C.sizeof_struct_sockaddr_in6
	sizeofInet6Pktinfo          = C.sizeof_struct_in6_pktinfo
	sizeofIPv6Mtuinfo           = C.sizeof_struct_ip6_mtuinfo
	sizeofIPv6FlowlabelReq      = C.sizeof_struct_in6_flowlabel_req

	sizeofIPv6Mreq       = C.sizeof_struct_ipv6_mreq
	sizeofGroupReq       = C.sizeof_struct_group_req
	sizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sizeofICMPv6Filter = C.sizeof_struct_icmp6_filter
)

type kernelSockaddrStorage C.struct___kernel_sockaddr_storage

type sockaddrInet6 C.struct_sockaddr_in6

type inet6Pktinfo C.struct_in6_pktinfo

type ipv6Mtuinfo C.struct_ip6_mtuinfo

type ipv6FlowlabelReq C.struct_in6_flowlabel_req

type ipv6Mreq C.struct_ipv6_mreq

type groupReq C.struct_group_req

type groupSourceReq C.struct_group_source_req

type icmpv6Filter C.struct_icmp6_filter
