// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// +godefs map struct_in_addr [4]byte /* in_addr */

package ipv4

/*
#include <time.h>

#include <linux/errqueue.h>
#include <linux/icmp.h>
#include <linux/in.h>
#include <linux/filter.h>
#include <sys/socket.h>
*/
import "C"

const (
	sizeofKernelSockaddrStorage = C.sizeof_struct___kernel_sockaddr_storage
	sizeofSockaddrInet          = C.sizeof_struct_sockaddr_in
	sizeofInetPktinfo           = C.sizeof_struct_in_pktinfo
	sizeofSockExtendedErr       = C.sizeof_struct_sock_extended_err

	sizeofIPMreq         = C.sizeof_struct_ip_mreq
	sizeofIPMreqSource   = C.sizeof_struct_ip_mreq_source
	sizeofGroupReq       = C.sizeof_struct_group_req
	sizeofGroupSourceReq = C.sizeof_struct_group_source_req

	sizeofICMPFilter = C.sizeof_struct_icmp_filter
)

type kernelSockaddrStorage C.struct___kernel_sockaddr_storage

type sockaddrInet C.struct_sockaddr_in

type inetPktinfo C.struct_in_pktinfo

type sockExtendedErr C.struct_sock_extended_err

type ipMreq C.struct_ip_mreq

type ipMreqSource C.struct_ip_mreq_source

type groupReq C.struct_group_req

type groupSourceReq C.struct_group_source_req

type icmpFilter C.struct_icmp_filter
