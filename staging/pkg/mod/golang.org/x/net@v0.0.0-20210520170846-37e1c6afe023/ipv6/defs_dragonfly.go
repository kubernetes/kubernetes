// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
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
	sizeofSockaddrInet6 = C.sizeof_struct_sockaddr_in6
	sizeofInet6Pktinfo  = C.sizeof_struct_in6_pktinfo
	sizeofIPv6Mtuinfo   = C.sizeof_struct_ip6_mtuinfo

	sizeofIPv6Mreq = C.sizeof_struct_ipv6_mreq

	sizeofICMPv6Filter = C.sizeof_struct_icmp6_filter
)

type sockaddrInet6 C.struct_sockaddr_in6

type inet6Pktinfo C.struct_in6_pktinfo

type ipv6Mtuinfo C.struct_ip6_mtuinfo

type ipv6Mreq C.struct_ipv6_mreq

type icmpv6Filter C.struct_icmp6_filter
