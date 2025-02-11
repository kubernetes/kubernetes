/*
Copyright 2019 The Kubernetes Authors.

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

package proxy

import (
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

// CategorizeEndpoints returns:
//
//   - The service's usable Cluster-traffic-policy endpoints (taking topology into account, if
//     relevant). This will be nil if the service does not ever use Cluster traffic policy.
//
//   - The service's usable Local-traffic-policy endpoints. This will be nil if the
//     service does not ever use Local traffic policy.
//
//   - The combined list of all endpoints reachable from this node (which is the union of the
//     previous two lists, but in the case where it is identical to one or the other, we avoid
//     allocating a separate list).
//
//   - An indication of whether the service has any endpoints reachable from anywhere in the
//     cluster. (This may be true even if allReachableEndpoints is empty.)
//
// "Usable endpoints" means Ready endpoints by default, but will fall back to
// Serving-Terminating endpoints (independently for Cluster and Local) if no Ready
// endpoints are available.
func CategorizeEndpoints(endpoints []Endpoint, svcInfo ServicePort, nodeName string, nodeLabels map[string]string) (clusterEndpoints, localEndpoints, allReachableEndpoints []Endpoint, hasAnyEndpoints bool) {
	var topologyMode string
	var useServingTerminatingEndpoints bool

	if svcInfo.UsesClusterEndpoints() {
		zone := nodeLabels[v1.LabelTopologyZone]
		topologyMode = topologyModeFromHints(svcInfo, endpoints, nodeName, zone)
		clusterEndpoints = filterEndpoints(endpoints, func(ep Endpoint) bool {
			if !ep.IsReady() {
				return false
			}
			if !availableForTopology(ep, topologyMode, nodeName, zone) {
				return false
			}
			return true
		})

		// If we didn't get any endpoints, try again using terminating endpoints.
		// (Note that we would already have chosen to ignore topology if there
		// were no ready endpoints for the given topology, so the problem at this
		// point must be that there are no ready endpoints anywhere.)
		if len(clusterEndpoints) == 0 {
			clusterEndpoints = filterEndpoints(endpoints, func(ep Endpoint) bool {
				if ep.IsServing() && ep.IsTerminating() {
					return true
				}

				return false
			})
		}

		// If there are any Ready endpoints anywhere in the cluster, we are
		// guaranteed to get one in clusterEndpoints.
		if len(clusterEndpoints) > 0 {
			hasAnyEndpoints = true
		}
	}

	if !svcInfo.UsesLocalEndpoints() {
		allReachableEndpoints = clusterEndpoints
		return
	}

	// Pre-scan the endpoints, to figure out which type of endpoint Local
	// traffic policy will use, and also to see if there are any usable
	// endpoints anywhere in the cluster.
	var hasLocalReadyEndpoints, hasLocalServingTerminatingEndpoints bool
	for _, ep := range endpoints {
		if ep.IsReady() {
			hasAnyEndpoints = true
			if ep.IsLocal() {
				hasLocalReadyEndpoints = true
			}
		} else if ep.IsServing() && ep.IsTerminating() {
			hasAnyEndpoints = true
			if ep.IsLocal() {
				hasLocalServingTerminatingEndpoints = true
			}
		}
	}

	if hasLocalReadyEndpoints {
		localEndpoints = filterEndpoints(endpoints, func(ep Endpoint) bool {
			return ep.IsLocal() && ep.IsReady()
		})
	} else if hasLocalServingTerminatingEndpoints {
		useServingTerminatingEndpoints = true
		localEndpoints = filterEndpoints(endpoints, func(ep Endpoint) bool {
			return ep.IsLocal() && ep.IsServing() && ep.IsTerminating()
		})
	}

	if !svcInfo.UsesClusterEndpoints() {
		allReachableEndpoints = localEndpoints
		return
	}

	if topologyMode == "" && !useServingTerminatingEndpoints {
		// !useServingTerminatingEndpoints means that localEndpoints contains only
		// Ready endpoints. topologyMode=="" means that clusterEndpoints contains *every*
		// Ready endpoint. So clusterEndpoints must be a superset of localEndpoints.
		allReachableEndpoints = clusterEndpoints
		return
	}

	// clusterEndpoints may contain remote endpoints that aren't in localEndpoints, while
	// localEndpoints may contain terminating or topologically-unavailable local endpoints
	// that aren't in clusterEndpoints. So we have to merge the two lists.
	endpointsMap := make(map[string]Endpoint, len(clusterEndpoints)+len(localEndpoints))
	for _, ep := range clusterEndpoints {
		endpointsMap[ep.String()] = ep
	}
	for _, ep := range localEndpoints {
		endpointsMap[ep.String()] = ep
	}
	allReachableEndpoints = make([]Endpoint, 0, len(endpointsMap))
	for _, ep := range endpointsMap {
		allReachableEndpoints = append(allReachableEndpoints, ep)
	}

	return
}

// topologyModeFromHints returns a topology mode ("", "PreferSameZone", or
// "PreferSameNode") based on the Endpoint hints:
//   - If the PreferSameTrafficDistribution feature gate is enabled, and every ready
//     endpoint has a node hint, and at least one endpoint is hinted for this node, then
//     it returns "PreferSameNode".
//   - Otherwise, if every ready endpoint has a zone hint, and at least one endpoint is
//     hinted for this node's zone, then it returns "PreferSameZone".
//   - Otherwise it returns "" (meaning, no topology / default traffic distribution).
func topologyModeFromHints(svcInfo ServicePort, endpoints []Endpoint, nodeName, zone string) string {
	hasEndpointForNode := false
	allEndpointsHaveNodeHints := true
	hasEndpointForZone := false
	allEndpointsHaveZoneHints := true
	for _, endpoint := range endpoints {
		if !endpoint.IsReady() {
			continue
		}

		if endpoint.NodeHints().Len() == 0 {
			allEndpointsHaveNodeHints = false
		} else if endpoint.NodeHints().Has(nodeName) {
			hasEndpointForNode = true
		}

		if endpoint.ZoneHints().Len() == 0 {
			allEndpointsHaveZoneHints = false
		} else if endpoint.ZoneHints().Has(zone) {
			hasEndpointForZone = true
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.PreferSameTrafficDistribution) {
		if allEndpointsHaveNodeHints {
			if hasEndpointForNode {
				return v1.ServiceTrafficDistributionPreferSameNode
			}
			klog.V(2).InfoS("Ignoring same-node topology hints for service since no hints were provided for node", "service", svcInfo, "node", nodeName)
		} else {
			klog.V(7).InfoS("Ignoring same-node topology hints for service since one or more endpoints is missing a node hint", "service", svcInfo)
		}
	}
	if allEndpointsHaveZoneHints {
		if hasEndpointForZone {
			return v1.ServiceTrafficDistributionPreferSameZone
		}
		if zone == "" {
			klog.V(2).InfoS("Ignoring same-zone topology hints for service since node is missing label", "service", svcInfo, "label", v1.LabelTopologyZone)
		} else {
			klog.V(2).InfoS("Ignoring same-zone topology hints for service since no hints were provided for zone", "service", svcInfo, "zone", zone)
		}
	} else {
		klog.V(7).InfoS("Ignoring same-zone topology hints for service since one or more endpoints is missing a zone hint", "service", svcInfo.String())
	}

	return ""
}

// availableForTopology checks if this endpoint is available for use on this node when
// using the given topologyMode. (Note that there's no fallback here; the fallback happens
// when deciding which mode to use, not when applying that decision.)
func availableForTopology(endpoint Endpoint, topologyMode, nodeName, zone string) bool {
	switch topologyMode {
	case "":
		return true
	case v1.ServiceTrafficDistributionPreferSameNode:
		return endpoint.NodeHints().Has(nodeName)
	case v1.ServiceTrafficDistributionPreferSameZone:
		return endpoint.ZoneHints().Has(zone)
	default:
		return false
	}
}

// filterEndpoints filters endpoints according to predicate
func filterEndpoints(endpoints []Endpoint, predicate func(Endpoint) bool) []Endpoint {
	filteredEndpoints := make([]Endpoint, 0, len(endpoints))

	for _, ep := range endpoints {
		if predicate(ep) {
			filteredEndpoints = append(filteredEndpoints, ep)
		}
	}

	return filteredEndpoints
}
