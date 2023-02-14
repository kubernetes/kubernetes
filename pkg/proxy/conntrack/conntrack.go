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
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/utils/exec"
	utilnet "k8s.io/utils/net"
	"strings"
)

// Interface is an injectable interface for cleaning connection tracking entries.
type Interface interface {
	CleanUp(proxy.ServicePortMap, proxy.UpdateServiceMapResult, proxy.UpdateEndpointMapResult)
}

type Conntrack struct {
	execer exec.Interface
}

// NewConntrack returns Conntrack which proxy backends can use to clear up stale connection
// entries.
func NewConntrack(execer exec.Interface) *Conntrack {
	return &Conntrack{
		execer: execer,
	}
}

// CleanUp understands kube-proxy logic and clears entries by iterating over proxy.UpdateServiceMapResult
// and proxy.UpdateEndpointMapResult after proxier sync call.
func (c *Conntrack) CleanUp(svcPortMap proxy.ServicePortMap, serviceUpdateResult proxy.UpdateServiceMapResult, endpointUpdateResult proxy.UpdateEndpointMapResult) {
	// clear conntrack entries for UDP services and service node ports.
	c.CleanServiceConnections(svcPortMap, serviceUpdateResult, endpointUpdateResult)
	// clear conntrack entries for stale UDP and SCTP services and service node ports.
	c.CleanEndpointConnections(svcPortMap, endpointUpdateResult)

}

// CleanServiceConnections clears conntrack entries for UDP services and service node ports.
func (c *Conntrack) CleanServiceConnections(svcPortMap proxy.ServicePortMap, serviceUpdateResult proxy.UpdateServiceMapResult, endpointUpdateResult proxy.UpdateEndpointMapResult) {
	conntrackCleanupServiceIPs := serviceUpdateResult.UDPStaleClusterIP

	for _, svcPortName := range endpointUpdateResult.StaleServiceNames {
		if svcInfo, ok := svcPortMap[svcPortName]; ok && svcInfo != nil && IsClearConntrackNeeded(svcInfo.Protocol()) {
			klog.V(2).InfoS("Stale service", "protocol", strings.ToLower(string(svcInfo.Protocol())), "servicePortName", svcPortName, "clusterIP", svcInfo.ClusterIP())
			conntrackCleanupServiceIPs.Insert(svcInfo.ClusterIP().String())

			for _, extIP := range svcInfo.ExternalIPStrings() {
				conntrackCleanupServiceIPs.Insert(extIP)
			}
			for _, lbIP := range svcInfo.LoadBalancerIPStrings() {
				conntrackCleanupServiceIPs.Insert(lbIP)
			}
			nodePort := svcInfo.NodePort()
			if svcInfo.Protocol() == v1.ProtocolUDP && nodePort != 0 {
				klog.V(2).InfoS("Stale service", "protocol", strings.ToLower(string(svcInfo.Protocol())), "servicePortName", svcPortName, "nodePort", nodePort)
				err := ClearEntriesForPort(c.execer, nodePort, utilnet.IsIPv6String(svcInfo.ClusterIP().String()), v1.ProtocolUDP)
				if err != nil {
					klog.ErrorS(err, "Failed to clear udp conntrack", "nodePort", nodePort)
				}
			}
		}
	}

	klog.V(4).InfoS("Deleting conntrack stale entries for services", "IPs", conntrackCleanupServiceIPs.UnsortedList())
	for _, svcIP := range conntrackCleanupServiceIPs.UnsortedList() {
		if err := ClearEntriesForIP(c.execer, svcIP, v1.ProtocolUDP); err != nil {
			klog.ErrorS(err, "Failed to delete stale service connections", "IP", svcIP)
		}
	}

}

// CleanEndpointConnections clears conntrack entries for stale UDP and SCTP services and service node ports.
func (c *Conntrack) CleanEndpointConnections(svcPortMap proxy.ServicePortMap, endpointUpdateResult proxy.UpdateEndpointMapResult) {
	// iterate over stale endpoints and clear conntrack entries
	klog.V(4).InfoS("Deleting stale endpoint connections", "endpoints", endpointUpdateResult.StaleEndpoints)
	for _, epSvcPair := range endpointUpdateResult.StaleEndpoints {
		if svcInfo, ok := svcPortMap[epSvcPair.ServicePortName]; ok && IsClearConntrackNeeded(svcInfo.Protocol()) {
			endpointIP := utilproxy.IPPart(epSvcPair.Endpoint)
			nodePort := svcInfo.NodePort()
			svcProto := svcInfo.Protocol()
			var err error
			if nodePort != 0 {
				err = ClearEntriesForPortNAT(c.execer, endpointIP, nodePort, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete nodeport-related endpoint connections", "servicePortName", epSvcPair.ServicePortName)
				}
			}
			err = ClearEntriesForNAT(c.execer, svcInfo.ClusterIP().String(), endpointIP, svcProto)
			if err != nil {
				klog.ErrorS(err, "Failed to delete endpoint connections", "servicePortName", epSvcPair.ServicePortName)
			}
			for _, extIP := range svcInfo.ExternalIPStrings() {
				err := ClearEntriesForNAT(c.execer, extIP, endpointIP, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete endpoint connections for externalIP", "servicePortName", epSvcPair.ServicePortName, "externalIP", extIP)
				}
			}
			for _, lbIP := range svcInfo.LoadBalancerIPStrings() {
				err := ClearEntriesForNAT(c.execer, lbIP, endpointIP, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete endpoint connections for LoadBalancerIP", "servicePortName", epSvcPair.ServicePortName, "loadBalancerIP", lbIP)
				}
			}
		}
	}
}
