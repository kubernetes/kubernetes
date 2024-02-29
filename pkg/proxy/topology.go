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
//   - The service's usable Local-traffic-policy endpoints (including terminating endpoints, if
//     relevant). This will be nil if the service does not ever use Local traffic policy.
//
//   - The combined list of all endpoints reachable from this node (which is the union of the
//     previous two lists, but in the case where it is identical to one or the other, we avoid
//     allocating a separate list).
//
//   - An indication of whether the service has any endpoints reachable from anywhere in the
//     cluster. (This may be true even if allReachableEndpoints is empty.)
func CategorizeEndpoints(endpoints []Endpoint, svcInfo ServicePort, nodeLabels map[string]string) (clusterEndpoints, localEndpoints, allReachableEndpoints []Endpoint, hasAnyEndpoints bool) {
	var useTopology, useServingTerminatingEndpoints bool

	if svcInfo.UsesClusterEndpoints() {
		useTopology = canUseTopology(endpoints, svcInfo, nodeLabels)
		clusterEndpoints = filterEndpoints(endpoints, func(ep Endpoint) bool {
			if !ep.IsReady() {
				return false
			}
			if useTopology && !availableForTopology(ep, nodeLabels) {
				return false
			}
			return true
		})

		// if there are 0 cluster-wide endpoints, we can try to fallback to any terminating endpoints that are ready.
		// When falling back to terminating endpoints, we do NOT consider topology aware routing since this is a best
		// effort attempt to avoid dropping connections.
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

	if !useTopology && !useServingTerminatingEndpoints {
		// !useServingTerminatingEndpoints means that localEndpoints contains only
		// Ready endpoints. !useTopology means that clusterEndpoints contains *every*
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

// canUseTopology returns true if topology aware routing is enabled and properly configured
// in this cluster. That is, it checks that:
// * The TopologyAwareHints feature is enabled
// * The "service.kubernetes.io/topology-aware-hints" annotation on this Service is set to "Auto"
// * The node's labels include "topology.kubernetes.io/zone"
// * All of the endpoints for this Service have a topology hint
// * At least one endpoint for this Service is hinted for this node's zone.
func canUseTopology(endpoints []Endpoint, svcInfo ServicePort, nodeLabels map[string]string) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareHints) {
		return false
	}
	// Any non-empty and non-disabled values for the hints annotation are acceptable.
	hintsAnnotation := svcInfo.HintsAnnotation()
	if hintsAnnotation == "" || hintsAnnotation == "disabled" || hintsAnnotation == "Disabled" {
		return false
	}

	zone, ok := nodeLabels[v1.LabelTopologyZone]
	if !ok || zone == "" {
		klog.V(2).InfoS("Skipping topology aware endpoint filtering since node is missing label", "label", v1.LabelTopologyZone)
		return false
	}

	hasEndpointForZone := false
	for _, endpoint := range endpoints {
		if !endpoint.IsReady() {
			continue
		}
		if endpoint.ZoneHints().Len() == 0 {
			klog.V(2).InfoS("Skipping topology aware endpoint filtering since one or more endpoints is missing a zone hint", "endpoint", endpoint)
			return false
		}

		if endpoint.ZoneHints().Has(zone) {
			hasEndpointForZone = true
		}
	}

	if !hasEndpointForZone {
		klog.V(2).InfoS("Skipping topology aware endpoint filtering since no hints were provided for zone", "zone", zone)
		return false
	}

	return true
}

// availableForTopology checks if this endpoint is available for use on this node, given
// topology constraints. (It assumes that canUseTopology() returned true.)
func availableForTopology(endpoint Endpoint, nodeLabels map[string]string) bool {
	zone := nodeLabels[v1.LabelTopologyZone]
	return endpoint.ZoneHints().Has(zone)
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
