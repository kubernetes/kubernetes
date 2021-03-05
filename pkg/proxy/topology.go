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

// FilterEndpoints filters endpoints based on Service configuration, node
// labels, and enabled feature gates. This is primarily used to enable topology
// aware routing.
func FilterEndpoints(endpoints []Endpoint, svcInfo ServicePort, nodeLabels map[string]string) []Endpoint {
	if svcInfo.NodeLocalExternal() || !utilfeature.DefaultFeatureGate.Enabled(features.EndpointSliceProxying) {
		return endpoints
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceTopology) {
		return deprecatedTopologyFilter(nodeLabels, svcInfo.TopologyKeys(), endpoints)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceInternalTrafficPolicy) && svcInfo.NodeLocalInternal() {
		return filterEndpointsInternalTrafficPolicy(svcInfo.InternalTrafficPolicy(), endpoints)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareHints) {
		return filterEndpointsWithHints(endpoints, svcInfo.HintsAnnotation(), nodeLabels)
	}

	return endpoints
}

// filterEndpointsWithHints provides filtering based on the hints included in
// EndpointSlices. If any of the following are true, the full list of endpoints
// will be returned without any filtering:
// * The AnnotationTopologyAwareHints annotation is not set to "auto" for this
//   Service.
// * No zone is specified in node labels.
// * No endpoints for this Service have a hint pointing to the zone this
//   instance of kube-proxy is running in.
// * One or more endpoints for this Service do not have hints specified.
func filterEndpointsWithHints(endpoints []Endpoint, hintsAnnotation string, nodeLabels map[string]string) []Endpoint {
	if hintsAnnotation != "auto" {
		if hintsAnnotation != "" && hintsAnnotation != "disabled" {
			klog.Warningf("Skipping topology aware endpoint filtering since Service has unexpected value for %s annotation: %s", v1.AnnotationTopologyAwareHints, hintsAnnotation)
		}
		return endpoints
	}

	zone, ok := nodeLabels[v1.LabelTopologyZone]
	if !ok || zone == "" {
		klog.Warningf("Skipping topology aware endpoint filtering since node is missing %s label", v1.LabelTopologyZone)
		return endpoints
	}

	filteredEndpoints := []Endpoint{}

	for _, endpoint := range endpoints {
		if endpoint.GetZoneHints().Len() == 0 {
			klog.Warningf("Skipping topology aware endpoint filtering since one or more endpoints is missing a zone hint")
			return endpoints
		}
		if endpoint.GetZoneHints().Has(zone) {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	if len(filteredEndpoints) > 0 {
		klog.Warningf("Skipping topology aware endpoint filtering since no hints were provided for zone %s", zone)
		return filteredEndpoints
	}

	return endpoints
}

// deprecatedTopologyFilter returns the appropriate endpoints based on the
// cluster topology. This will be removed in an upcoming release along with the
// ServiceTopology feature gate.
//
// This uses the current node's labels, which contain topology information, and
// the required topologyKeys to find appropriate endpoints. If both the endpoint's
// topology and the current node have matching values for topologyKeys[0], the
// endpoint will be chosen.  If no endpoints are chosen, toplogyKeys[1] will be
// considered, and so on.  If either the node or the endpoint do not have values
// for a key, it is considered to not match.
//
// If topologyKeys is specified, but no endpoints are chosen for any key, the
// service has no viable endpoints for clients on this node, and connections
// should fail.
//
// The special key "*" may be used as the last entry in topologyKeys to indicate
// "any endpoint" is acceptable.
//
// If topologyKeys is not specified or empty, no topology constraints will be
// applied and this will return all endpoints.
func deprecatedTopologyFilter(nodeLabels map[string]string, topologyKeys []string, endpoints []Endpoint) []Endpoint {
	// Do not filter endpoints if service has no topology keys.
	if len(topologyKeys) == 0 {
		return endpoints
	}

	filteredEndpoints := []Endpoint{}

	if len(nodeLabels) == 0 {
		if topologyKeys[len(topologyKeys)-1] == v1.TopologyKeyAny {
			// edge case: include all endpoints if topology key "Any" specified
			// when we cannot determine current node's topology.
			return endpoints
		}
		// edge case: do not include any endpoints if topology key "Any" is
		// not specified when we cannot determine current node's topology.
		return filteredEndpoints
	}

	for _, key := range topologyKeys {
		if key == v1.TopologyKeyAny {
			return endpoints
		}
		topologyValue, found := nodeLabels[key]
		if !found {
			continue
		}

		for _, ep := range endpoints {
			topology := ep.GetTopology()
			if value, found := topology[key]; found && value == topologyValue {
				filteredEndpoints = append(filteredEndpoints, ep)
			}
		}
		if len(filteredEndpoints) > 0 {
			return filteredEndpoints
		}
	}
	return filteredEndpoints
}

// filterEndpointsInternalTrafficPolicy returns the node local endpoints based
// on configured InternalTrafficPolicy.
//
// If ServiceInternalTrafficPolicy feature gate is off, returns the original
// EndpointSlice.
// Otherwise, if InternalTrafficPolicy is Local, only return the node local endpoints.
func filterEndpointsInternalTrafficPolicy(internalTrafficPolicy *v1.ServiceInternalTrafficPolicyType, endpoints []Endpoint) []Endpoint {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceInternalTrafficPolicy) {
		return endpoints
	}
	if internalTrafficPolicy == nil || *internalTrafficPolicy == v1.ServiceInternalTrafficPolicyCluster {
		return endpoints
	}

	var filteredEndpoints []Endpoint

	// Get all the local endpoints
	for _, ep := range endpoints {
		if ep.GetIsLocal() {
			filteredEndpoints = append(filteredEndpoints, ep)
		}
	}

	// When internalTrafficPolicy is Local, only return the node local
	// endpoints
	return filteredEndpoints
}
