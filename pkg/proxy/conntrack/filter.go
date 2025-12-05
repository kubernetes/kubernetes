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
	"strconv"

	"github.com/vishvananda/netlink"

	"k8s.io/apimachinery/pkg/util/sets"
)

type conntrackFilter struct {
	protocol uint8
	// serviceIPEndpoints maps service IPs (ClusterIP, LoadBalancerIPs and ExternalIPs) and Service Port
	// to the set of serving endpoints (Endpoint IP and Port).
	serviceIPEndpoints map[string]sets.Set[string]
	// serviceNodePortEndpoints maps service NodePort to the set of serving endpoints  (Endpoint IP and Port).
	serviceNodePortEndpoints map[int]sets.Set[string]
}

var _ netlink.CustomConntrackFilter = (*conntrackFilter)(nil)

// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches the filter
// false otherwise.
func (f *conntrackFilter) MatchConntrackFlow(flow *netlink.ConntrackFlow) bool {
	// filter out the protocol
	if flow.Forward.Protocol != f.protocol {
		return false
	}

	origDst := flow.Forward.DstIP.String()   // match Service IP
	origPortDst := int(flow.Forward.DstPort) // match Service Port
	origPortDstStr := strconv.Itoa(origPortDst)
	replySrc := flow.Reverse.SrcIP.String()   // match Serving Endpoint IP
	replyPortSrc := int(flow.Reverse.SrcPort) // match Serving Endpoint Port
	replyPortSrcStr := strconv.Itoa(replyPortSrc)

	// if the original destination (--orig-dst) of the entry is service IP (ClusterIP,
	// LoadBalancerIPs or ExternalIPs) and (--orig-port-dst) is service Port and
	// the reply source IP (--reply-src) and port (--reply-port-src) does not
	// represent a serving endpoint of the service, we clear the entry.
	endpoints, ok := f.serviceIPEndpoints[net.JoinHostPort(origDst, origPortDstStr)]
	if ok && !endpoints.Has(net.JoinHostPort(replySrc, replyPortSrcStr)) {
		return true
	}

	// if the original destination port (--orig-port-dst) of the entry is service
	// NodePort and the reply source IP (--reply-src) and port (--reply-port-src)
	// does not represent a serving endpoint of the service, we clear the entry.
	endpoints, ok = f.serviceNodePortEndpoints[origPortDst]
	if ok && !endpoints.Has(net.JoinHostPort(replySrc, replyPortSrcStr)) {
		return true
	}

	return false
}
