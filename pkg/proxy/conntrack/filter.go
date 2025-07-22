//go:build linux
// +build linux

/*
Copyright 2024 The Kubernetes Authors.

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
	"net"

	"github.com/vishvananda/netlink"

	"k8s.io/klog/v2"
)

type connectionTuple struct {
	srcIP   net.IP
	srcPort uint16
	dstIP   net.IP
	dstPort uint16
}

type conntrackFilter struct {
	protocol uint8
	original *connectionTuple
	reply    *connectionTuple
}

var _ netlink.CustomConntrackFilter = (*conntrackFilter)(nil)

// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches the filter
// false otherwise.
func (f *conntrackFilter) MatchConntrackFlow(flow *netlink.ConntrackFlow) bool {
	// return false in case of empty filter
	if f.protocol == 0 && f.original == nil && f.reply == nil {
		return false
	}

	// -p, --protonum proto [Layer 4 Protocol, eg. 'tcp']
	if f.protocol != 0 && f.protocol != flow.Forward.Protocol {
		return false
	}

	// filter on original direction
	if f.original != nil {
		// --orig-src ip  [Source address from original direction]
		if f.original.srcIP != nil && !f.original.srcIP.Equal(flow.Forward.SrcIP) {
			return false
		}
		// --orig-dst ip  [Destination address from original direction]
		if f.original.dstIP != nil && !f.original.dstIP.Equal(flow.Forward.DstIP) {
			return false
		}
		// --orig-port-src port [Source port from original direction]
		if f.original.srcPort != 0 && f.original.srcPort != flow.Forward.SrcPort {
			return false
		}
		// --orig-port-dst port	[Destination port from original direction]
		if f.original.dstPort != 0 && f.original.dstPort != flow.Forward.DstPort {
			return false
		}
	}

	// filter on reply direction
	if f.reply != nil {
		// --reply-src ip  [Source NAT ip]
		if f.reply.srcIP != nil && !f.reply.srcIP.Equal(flow.Reverse.SrcIP) {
			return false
		}
		// --reply-dst ip [Destination NAT ip]
		if f.reply.dstIP != nil && !f.reply.dstIP.Equal(flow.Reverse.DstIP) {
			return false
		}
		// --reply-port-src port [Source port from reply direction]
		if f.reply.srcPort != 0 && f.reply.srcPort != flow.Reverse.SrcPort {
			return false
		}
		// --reply-port-dst port	[Destination port from reply direction]
		if f.reply.dstPort != 0 && f.reply.dstPort != flow.Reverse.DstPort {
			return false
		}
	}

	// appending a new line to the flow makes klog print multiline log which is easier to debug and understand.
	klog.V(5).InfoS("Deleting conntrack entry", "flow", flow.String()+"\n")
	return true
}
