// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

// Sticky socket options
const (
	ssoTrafficClass        = iota // header field for unicast packet, RFC 3542
	ssoHopLimit                   // header field for unicast packet, RFC 3493
	ssoMulticastInterface         // outbound interface for multicast packet, RFC 3493
	ssoMulticastHopLimit          // header field for multicast packet, RFC 3493
	ssoMulticastLoopback          // loopback for multicast packet, RFC 3493
	ssoReceiveTrafficClass        // header field on received packet, RFC 3542
	ssoReceiveHopLimit            // header field on received packet, RFC 2292 or 3542
	ssoReceivePacketInfo          // incbound or outbound packet path, RFC 2292 or 3542
	ssoReceivePathMTU             // path mtu, RFC 3542
	ssoPathMTU                    // path mtu, RFC 3542
	ssoChecksum                   // packet checksum, RFC 2292 or 3542
	ssoICMPFilter                 // icmp filter, RFC 2292 or 3542
	ssoJoinGroup                  // any-source multicast, RFC 3493
	ssoLeaveGroup                 // any-source multicast, RFC 3493
	ssoJoinSourceGroup            // source-specific multicast
	ssoLeaveSourceGroup           // source-specific multicast
	ssoBlockSourceGroup           // any-source or source-specific multicast
	ssoUnblockSourceGroup         // any-source or source-specific multicast
	ssoMax
)

// Sticky socket option value types
const (
	ssoTypeInt = iota + 1
	ssoTypeInterface
	ssoTypeICMPFilter
	ssoTypeMTUInfo
	ssoTypeIPMreq
	ssoTypeGroupReq
	ssoTypeGroupSourceReq
)

// A sockOpt represents a binding for sticky socket option.
type sockOpt struct {
	level int // option level
	name  int // option name, must be equal or greater than 1
	typ   int // option value type, must be equal or greater than 1
}
