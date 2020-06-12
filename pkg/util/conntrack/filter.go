/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package conntrack

import (
	"errors"
	"net"

	"github.com/vishvananda/netlink"
)

// Add custom Layer 4 protocol filter based on:
// https://github.com/vishvananda/netlink/blob/214c19c0cf881d37cf772dea0d6c485bbbecdc26/conntrack_linux.go

// Conntrack parameters and options:
//   -n, --src-nat ip                      source NAT ip
//   -g, --dst-nat ip                      destination NAT ip
//   -j, --any-nat ip                      source or destination NAT ip
//   -m, --mark mark                       Set mark
//   -c, --secmark secmark                 Set selinux secmark
//   -e, --event-mask eventmask            Event mask, eg. NEW,DESTROY
//   -z, --zero                            Zero counters while listing
//   -o, --output type[,...]               Output format, eg. xml
//   -l, --label label[,...]               conntrack labels

// Common parameters and options:
//   -s, --src, --orig-src ip              Source address from original direction
//   -d, --dst, --orig-dst ip              Destination address from original direction
//   -r, --reply-src ip            Source address from reply direction
//   -q, --reply-dst ip            Destination address from reply direction
//   -p, --protonum proto          Layer 4 Protocol, eg. 'tcp'
//   -f, --family proto            Layer 3 Protocol, eg. 'ipv6'
//   -t, --timeout timeout         Set timeout
//   -u, --status status           Set status, eg. ASSURED
//   -w, --zone value              Set conntrack zone
//   --orig-zone value             Set zone for original direction
//   --reply-zone value            Set zone for reply direction
//   -b, --buffer-size             Netlink socket buffer size
//   --mask-src ip                 Source mask address
//   --mask-dst ip                 Destination mask address

// Layer 4 Protocol common parameters and options:
// TCP, UDP, SCTP, UDPLite and DCCP
//    --sport, --orig-port-src port    Source port in original direction
//    --dport, --orig-port-dst port    Destination port in original direction

// Filter types
type conntrackFilterType uint8

const (
	conntrackOrigSrcIP   = iota // -orig-src ip    Source address from original direction
	conntrackOrigDstIP          // -orig-dst ip    Destination address from original direction
	conntrackReplySrcIP         // --reply-src ip  Reply Source IP
	conntrackReplyDstIP         // --reply-dst ip  Reply Destination IP
	conntrackReplyAnyIP         // Match source or destination reply IP
	conntrackOrigSrcPort        // --orig-port-src port    Source port in original direction
	conntrackOrigDstPort        // --orig-port-dst port    Destination port in original direction
)

type conntrackFilter struct {
	ipFilter    map[conntrackFilterType]net.IP
	portFilter  map[conntrackFilterType]uint16
	protoFilter uint8
}

// AddIP adds an IP to the conntrack filter
func (f *conntrackFilter) addIP(tp conntrackFilterType, ip net.IP) error {
	if f.ipFilter == nil {
		f.ipFilter = make(map[conntrackFilterType]net.IP)
	}
	if _, ok := f.ipFilter[tp]; ok {
		return errors.New("Filter attribute already present")
	}
	f.ipFilter[tp] = ip
	return nil
}

// AddPort adds a Port to the conntrack filter if the Layer 4 protocol allows it
func (f *conntrackFilter) addPort(tp conntrackFilterType, port uint16) error {
	switch f.protoFilter {
	// TCP, UDP, DCCP, SCTP, UDPLite
	case 6, 17, 33, 132, 136:
	default:
		return errors.New("Filter attribute not available without a Layer 4 protocol")
	}

	if f.portFilter == nil {
		f.portFilter = make(map[conntrackFilterType]uint16)
	}
	if _, ok := f.portFilter[tp]; ok {
		return errors.New("Filter attribute already present")
	}
	f.portFilter[tp] = port
	return nil
}

// AddProtocol adds the Layer 4 protocol to the conntrack filter
func (f *conntrackFilter) addProtocol(proto uint8) error {
	if f.protoFilter != 0 {
		return errors.New("Filter attribute already present")
	}
	f.protoFilter = proto
	return nil
}

// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches the filter
// false otherwise
func (f *conntrackFilter) MatchConntrackFlow(flow *netlink.ConntrackFlow) bool {
	if len(f.ipFilter) == 0 && len(f.portFilter) == 0 && f.protoFilter == 0 {
		// empty filter always not match
		return false
	}

	// -p, --protonum proto          Layer 4 Protocol, eg. 'tcp'
	if f.protoFilter != 0 && flow.Forward.Protocol != f.protoFilter {
		// different Layer 4 protocol always not match
		return false
	}

	match := true

	// IP conntrack filter
	if len(f.ipFilter) > 0 {

		// -orig-src ip   Source address from original direction
		if elem, found := f.ipFilter[conntrackOrigSrcIP]; found {
			match = match && elem.Equal(flow.Forward.SrcIP)
		}

		// -orig-dst ip   Destination address from original direction
		if elem, found := f.ipFilter[conntrackOrigDstIP]; match && found {
			match = match && elem.Equal(flow.Forward.DstIP)
		}

		// -src-nat ip    Source NAT ip
		if elem, found := f.ipFilter[conntrackReplySrcIP]; match && found {
			match = match && elem.Equal(flow.Reverse.SrcIP)
		}

		// -dst-nat ip    Destination NAT ip
		if elem, found := f.ipFilter[conntrackReplyDstIP]; match && found {
			match = match && elem.Equal(flow.Reverse.DstIP)
		}

		// Match source or destination reply IP
		if elem, found := f.ipFilter[conntrackReplyAnyIP]; match && found {
			match = match && (elem.Equal(flow.Reverse.SrcIP) || elem.Equal(flow.Reverse.DstIP))
		}
	}

	// Layer 4 Port filter
	if len(f.portFilter) > 0 {

		// -orig-port-src port	Source port from original direction
		if elem, found := f.portFilter[conntrackOrigSrcPort]; match && found {
			match = match && elem == flow.Forward.SrcPort
		}

		// -orig-port-dst port	Destination port from original direction
		if elem, found := f.portFilter[conntrackOrigDstPort]; match && found {
			match = match && elem == flow.Forward.DstPort
		}
	}

	return match
}

var _ netlink.CustomConntrackFilter = (*conntrackFilter)(nil)
