//go:build linux
// +build linux

/*
Copyright 2023 The Kubernetes Authors.

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
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	netutils "k8s.io/utils/net"
)

// CleanStaleEntries takes care of flushing stale conntrack entries for services and endpoints.
func CleanStaleEntries(ct Interface, svcPortMap proxy.ServicePortMap,
	serviceUpdateResult proxy.UpdateServiceMapResult, endpointsUpdateResult proxy.UpdateEndpointsMapResult) {
	deleteStaleServiceConntrackEntries(ct, svcPortMap, serviceUpdateResult, endpointsUpdateResult)
	deleteStaleEndpointConntrackEntries(ct, svcPortMap, endpointsUpdateResult)
}

// deleteStaleServiceConntrackEntries takes care of flushing stale conntrack entries related
// to UDP Service IPs. When a service has no endpoints and we drop traffic to it, conntrack
// may create "black hole" entries for that IP+port. When the service gets endpoints we
// need to delete those entries so further traffic doesn't get dropped.
func deleteStaleServiceConntrackEntries(ct Interface, svcPortMap proxy.ServicePortMap, serviceUpdateResult proxy.UpdateServiceMapResult, endpointsUpdateResult proxy.UpdateEndpointsMapResult) {
	var filters []netlink.CustomConntrackFilter
	conntrackCleanupServiceIPs := serviceUpdateResult.DeletedUDPClusterIPs
	conntrackCleanupServiceNodePorts := sets.New[int]()
	isIPv6 := false

	// merge newly active services gathered from endpointsUpdateResult
	// a UDP service that changes from 0 to non-0 endpoints is newly active.
	for _, svcPortName := range endpointsUpdateResult.NewlyActiveUDPServices {
		if svcInfo, ok := svcPortMap[svcPortName]; ok {
			isIPv6 = netutils.IsIPv6(svcInfo.ClusterIP())
			klog.V(4).InfoS("Newly-active UDP service may have stale conntrack entries", "servicePortName", svcPortName)
			conntrackCleanupServiceIPs.Insert(svcInfo.ClusterIP().String())
			for _, extIP := range svcInfo.ExternalIPs() {
				conntrackCleanupServiceIPs.Insert(extIP.String())
			}
			for _, lbIP := range svcInfo.LoadBalancerVIPs() {
				conntrackCleanupServiceIPs.Insert(lbIP.String())
			}
			nodePort := svcInfo.NodePort()
			if svcInfo.Protocol() == v1.ProtocolUDP && nodePort != 0 {
				conntrackCleanupServiceNodePorts.Insert(nodePort)
			}
		}
	}

	klog.V(4).InfoS("Deleting conntrack stale entries for services", "IPs", conntrackCleanupServiceIPs.UnsortedList())
	for _, svcIP := range conntrackCleanupServiceIPs.UnsortedList() {
		filters = append(filters, filterForIP(svcIP, v1.ProtocolUDP))
	}
	klog.V(4).InfoS("Deleting conntrack stale entries for services", "nodePorts", conntrackCleanupServiceNodePorts.UnsortedList())
	for _, nodePort := range conntrackCleanupServiceNodePorts.UnsortedList() {
		filters = append(filters, filterForPort(nodePort, v1.ProtocolUDP))
	}

	if err := ct.ClearEntries(getUnixIPFamily(isIPv6), filters...); err != nil {
		klog.ErrorS(err, "Failed to delete stale service connections")
	}
}

// deleteStaleEndpointConntrackEntries takes care of flushing stale conntrack entries related
// to UDP endpoints. After a UDP endpoint is removed we must flush any conntrack entries
// for it so that if the same client keeps sending, the packets will get routed to a new endpoint.
func deleteStaleEndpointConntrackEntries(ct Interface, svcPortMap proxy.ServicePortMap, endpointsUpdateResult proxy.UpdateEndpointsMapResult) {
	var filters []netlink.CustomConntrackFilter
	isIPv6 := false
	for _, epSvcPair := range endpointsUpdateResult.DeletedUDPEndpoints {
		if svcInfo, ok := svcPortMap[epSvcPair.ServicePortName]; ok {
			isIPv6 = netutils.IsIPv6(svcInfo.ClusterIP())
			endpointIP := proxyutil.IPPart(epSvcPair.Endpoint)
			nodePort := svcInfo.NodePort()
			if nodePort != 0 {
				filters = append(filters, filterForPortNAT(endpointIP, nodePort, v1.ProtocolUDP))

			}
			filters = append(filters, filterForNAT(svcInfo.ClusterIP().String(), endpointIP, v1.ProtocolUDP))
			for _, extIP := range svcInfo.ExternalIPs() {
				filters = append(filters, filterForNAT(extIP.String(), endpointIP, v1.ProtocolUDP))
			}
			for _, lbIP := range svcInfo.LoadBalancerVIPs() {
				filters = append(filters, filterForNAT(lbIP.String(), endpointIP, v1.ProtocolUDP))
			}
		}
	}

	if err := ct.ClearEntries(getUnixIPFamily(isIPv6), filters...); err != nil {
		klog.ErrorS(err, "Failed to delete stale endpoint connections")
	}
}

// getUnixIPFamily returns the unix IPFamily constant.
func getUnixIPFamily(isIPv6 bool) uint8 {
	if isIPv6 {
		return unix.AF_INET6
	}
	return unix.AF_INET
}

// protocolMap maps v1.Protocol to the Assigned Internet Protocol Number.
// https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
var protocolMap = map[v1.Protocol]uint8{
	v1.ProtocolTCP:  unix.IPPROTO_TCP,
	v1.ProtocolUDP:  unix.IPPROTO_UDP,
	v1.ProtocolSCTP: unix.IPPROTO_SCTP,
}

// filterForIP returns *conntrackFilter to delete the conntrack entries for connections
// specified by the destination IP (original direction).
func filterForIP(ip string, protocol v1.Protocol) *conntrackFilter {
	klog.V(4).InfoS("Adding conntrack filter for cleanup", "org-dst", ip, "protocol", protocol)
	return &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstIP: netutils.ParseIPSloppy(ip),
		},
	}
}

// filterForPort returns *conntrackFilter to delete the conntrack entries for connections
// specified by the destination Port (original direction).
func filterForPort(port int, protocol v1.Protocol) *conntrackFilter {
	klog.V(4).InfoS("Adding conntrack filter for cleanup", "org-port-dst", port, "protocol", protocol)
	return &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstPort: uint16(port),
		},
	}
}

// filterForNAT returns *conntrackFilter to delete the conntrack entries for connections
// specified by the destination IP (original direction) and source IP (reply direction).
func filterForNAT(origin, dest string, protocol v1.Protocol) *conntrackFilter {
	klog.V(4).InfoS("Adding conntrack filter for cleanup", "org-dst", origin, "reply-src", dest, "protocol", protocol)
	return &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstIP: netutils.ParseIPSloppy(origin),
		},
		reply: &connectionTuple{
			srcIP: netutils.ParseIPSloppy(dest),
		},
	}
}

// filterForPortNAT returns *conntrackFilter to delete the conntrack entries for connections
// specified by the destination Port (original direction) and source IP (reply direction).
func filterForPortNAT(dest string, port int, protocol v1.Protocol) *conntrackFilter {
	klog.V(4).InfoS("Adding conntrack filter for cleanup", "org-port-dst", port, "reply-src", dest, "protocol", protocol)
	return &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstPort: uint16(port),
		},
		reply: &connectionTuple{
			srcIP: netutils.ParseIPSloppy(dest),
		},
	}
}
