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
	if svcInfo.NodeLocalExternal() {
		return endpoints
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceInternalTrafficPolicy) && svcInfo.NodeLocalInternal() {
		return FilterLocalEndpoint(endpoints)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareHints) {
		return filterEndpointsWithHints(endpoints, svcInfo.HintsAnnotation(), nodeLabels)
	}

	return endpoints
}

// filterEndpointsWithHints provides filtering based on the hints included in
// EndpointSlices. If any of the following are true, the full list of endpoints
// will be returned without any filtering:
// * The AnnotationTopologyAwareHints annotation is not set to "Auto" for this
//   Service.
// * No zone is specified in node labels.
// * No endpoints for this Service have a hint pointing to the zone this
//   instance of kube-proxy is running in.
// * One or more endpoints for this Service do not have hints specified.
func filterEndpointsWithHints(endpoints []Endpoint, hintsAnnotation string, nodeLabels map[string]string) []Endpoint {
	if hintsAnnotation != "Auto" && hintsAnnotation != "auto" {
		if hintsAnnotation != "" && hintsAnnotation != "Disabled" && hintsAnnotation != "disabled" {
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

	if len(filteredEndpoints) == 0 {
		klog.Warningf("Skipping topology aware endpoint filtering since no hints were provided for zone %s", zone)
		return endpoints
	}

	return filteredEndpoints
}

// FilterLocalEndpoint returns the node local endpoints
func FilterLocalEndpoint(endpoints []Endpoint) []Endpoint {
	var filteredEndpoints []Endpoint

	// Get all the local endpoints
	for _, ep := range endpoints {
		if ep.GetIsLocal() {
			filteredEndpoints = append(filteredEndpoints, ep)
		}
	}

	return filteredEndpoints
}
