// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

// Sticky socket options
const (
	ssoTOS                = iota // header field for unicast packet
	ssoTTL                       // header field for unicast packet
	ssoMulticastTTL              // header field for multicast packet
	ssoMulticastInterface        // outbound interface for multicast packet
	ssoMulticastLoopback         // loopback for multicast packet
	ssoReceiveTTL                // header field on received packet
	ssoReceiveDst                // header field on received packet
	ssoReceiveInterface          // inbound interface on received packet
	ssoPacketInfo                // incbound or outbound packet path
	ssoHeaderPrepend             // ipv4 header prepend
	ssoStripHeader               // strip ipv4 header
	ssoICMPFilter                // icmp filter
	ssoJoinGroup                 // any-source multicast
	ssoLeaveGroup                // any-source multicast
	ssoJoinSourceGroup           // source-specific multicast
	ssoLeaveSourceGroup          // source-specific multicast
	ssoBlockSourceGroup          // any-source or source-specific multicast
	ssoUnblockSourceGroup        // any-source or source-specific multicast
	ssoMax
)

// Sticky socket option value types
const (
	ssoTypeByte = iota + 1
	ssoTypeInt
	ssoTypeInterface
	ssoTypeICMPFilter
	ssoTypeIPMreq
	ssoTypeIPMreqn
	ssoTypeGroupReq
	ssoTypeGroupSourceReq
)

// A sockOpt represents a binding for sticky socket option.
type sockOpt struct {
	name int // option name, must be equal or greater than 1
	typ  int // option value type, must be equal or greater than 1
}
