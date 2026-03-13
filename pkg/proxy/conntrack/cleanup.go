//go:build linux

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
	"errors"
	"net"
	"strconv"
	"time"

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/metrics"
)

// Kubernetes UDP services can be affected by stale conntrack entries.
// These entries may point to endpoints that no longer exist,
// leading to packet loss and connectivity problems.

// CleanStaleEntries scans conntrack table and removes any entries
// for a service that do not correspond to a serving endpoint.
// List existing conntrack entries and calculate the desired conntrack state
// based on the current Services and Endpoints.
func CleanStaleEntries(ct Interface, ipFamily v1.IPFamily,
	svcPortMap proxy.ServicePortMap, endpointsMap proxy.EndpointsMap) {

	start := time.Now()
	klog.V(4).InfoS("Started to reconcile conntrack entries", "ipFamily", ipFamily)

	entries, err := ct.ListEntries(ipFamilyMap[ipFamily])
	if err != nil {
		if errors.Is(err, unix.EINTR) {
			klog.V(2).ErrorS(err, "received a partial result, continuing to clean with partial result")
		} else {
			klog.ErrorS(err, "Failed to list conntrack entries")
			return
		}
	}

	// serviceIPEndpoints maps service IPs (ClusterIP, LoadBalancerIPs and ExternalIPs) and Service Port
	// to the set of serving endpoints (Endpoint IP and Port).
	serviceIPEndpoints := make(map[string]sets.Set[string])
	// serviceNodePortEndpoints maps service NodePort to the set of serving endpoints  (Endpoint IP and Port).
	serviceNodePortEndpoints := make(map[int]sets.Set[string])

	for svcName, svc := range svcPortMap {
		// we are only interested in UDP services
		if svc.Protocol() != v1.ProtocolUDP {
			continue
		}

		endpoints := sets.New[string]()
		for _, endpoint := range endpointsMap[svcName] {
			// We need to remove all the conntrack entries for a Service (IP or NodePort)
			// that are not pointing to a serving endpoint.
			// We map all the serving endpoints of the service and clear all the conntrack
			// entries which are destined for the service and are not DNATed to these endpoints.
			// Changes to the service should not affect existing flows, so we do not take
			// traffic policies, topology, or terminating status of the service into account.
			// This ensures that the behavior of UDP services remains consistent with TCP
			// services.
			if endpoint.IsServing() {
				portStr := strconv.Itoa(int(endpoint.Port()))
				endpoints.Insert(net.JoinHostPort(endpoint.IP(), portStr))
			}
		}

		// a Service without endpoints does not require to clean the conntrack entries associated.
		if endpoints.Len() == 0 {
			continue
		}

		// we need to filter entries that are directed to a Service IP:Port frontend
		// that does not have an Endpoint IP:Port backend as part of the serving endpoints
		portStr := strconv.Itoa(svc.Port())
		// clusterIP:Port
		serviceIPEndpoints[net.JoinHostPort(svc.ClusterIP().String(), portStr)] = endpoints
		// loadbalancerIP:Port
		for _, loadBalancerIP := range svc.LoadBalancerVIPs() {
			serviceIPEndpoints[net.JoinHostPort(loadBalancerIP.String(), portStr)] = endpoints
		}
		// externalIP:Port
		for _, externalIP := range svc.ExternalIPs() {
			serviceIPEndpoints[net.JoinHostPort(externalIP.String(), portStr)] = endpoints
		}
		// we need to filter entries that are directed to a *:NodePort that does not have
		// an Endpoint IP:Port backend as part of the serving endpoints
		if svc.NodePort() != 0 {
			// *:NodePort
			serviceNodePortEndpoints[svc.NodePort()] = endpoints
		}
	}

	var flows []*netlink.ConntrackFlow
	for _, entry := range entries {
		// we only deal with UDP protocol entries
		if entry.Forward.Protocol != unix.IPPROTO_UDP {
			continue
		}

		origDst := entry.Forward.DstIP.String()   // match Service IP
		origPortDst := int(entry.Forward.DstPort) // match Service Port
		origPortDstStr := strconv.Itoa(origPortDst)
		replySrc := entry.Reverse.SrcIP.String()   // match Serving Endpoint IP
		replyPortSrc := int(entry.Reverse.SrcPort) // match Serving Endpoint Port
		replyPortSrcStr := strconv.Itoa(replyPortSrc)

		// if the original destination (--orig-dst) of the entry is service IP (ClusterIP,
		// LoadBalancerIPs or ExternalIPs) and (--orig-port-dst) is service Port and
		// the reply source IP (--reply-src) and port (--reply-port-src) does not
		// represent a serving endpoint of the service, we clear the entry.
		endpoints, ok := serviceIPEndpoints[net.JoinHostPort(origDst, origPortDstStr)]
		if ok && !endpoints.Has(net.JoinHostPort(replySrc, replyPortSrcStr)) {
			flows = append(flows, entry)
			continue
		}

		// if the original destination port (--orig-port-dst) of the entry is service
		// NodePort and the reply source IP (--reply-src) and port (--reply-port-src)
		// does not represent a serving endpoint of the service, we clear the entry.
		endpoints, ok = serviceNodePortEndpoints[origPortDst]
		if ok && !endpoints.Has(net.JoinHostPort(replySrc, replyPortSrcStr)) {
			flows = append(flows, entry)
			continue
		}
	}

	var n int
	if n, err = ct.DeleteEntries(ipFamilyMap[ipFamily], flows); err != nil {
		klog.ErrorS(err, "Failed to clear all conntrack entries", "ipFamily", ipFamily, "entriesDeleted", n, "took", time.Since(start))
	} else {
		klog.V(4).InfoS("Finished reconciling conntrack entries", "ipFamily", ipFamily, "entriesDeleted", n, "took", time.Since(start))
	}
	metrics.ReconcileConntrackFlowsLatency.WithLabelValues(string(ipFamily)).Observe(metrics.SinceInSeconds(start))
	metrics.ReconcileConntrackFlowsDeletedEntriesTotal.WithLabelValues(string(ipFamily)).Add(float64(n))
}

// ipFamilyMap maps v1.IPFamily to the corresponding unix constant.
var ipFamilyMap = map[v1.IPFamily]uint8{
	v1.IPv4Protocol: unix.AF_INET,
	v1.IPv6Protocol: unix.AF_INET6,
}
