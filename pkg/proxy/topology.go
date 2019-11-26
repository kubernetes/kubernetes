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
)

// FilterTopologyEndpoint returns the appropriate endpoints based on the cluster
// topology.
// This uses the current node's labels, which contain topology information, and
// the required topologyKeys to find appropriate endpoints. If both the endpoint's
// topology and the current node have matching values for topologyKeys[0], the
// endpoint will be chosen.  If no endpoints are chosen, toplogyKeys[1] will be
// considered, and so on.  If either the node or the endpoint do not have values
// for a key, it is considered to not match.
//
// If topologyKeys is specified, but no endpoints are chosen for any key, the
// the service has no viable endpoints for clients on this node, and connections
// should fail.
//
// The special key "*" may be used as the last entry in topologyKeys to indicate
// "any endpoint" is acceptable.
//
// If topologyKeys is not specified or empty, no topology constraints will be
// applied and this will return all endpoints.
func FilterTopologyEndpoint(nodeLabels map[string]string, topologyKeys []string, endpoints []Endpoint) []Endpoint {
	// Do not filter endpoints if service has no topology keys.
	if len(topologyKeys) == 0 {
		return endpoints
	}

	filteredEndpoint := []Endpoint{}

	if len(nodeLabels) == 0 {
		if topologyKeys[len(topologyKeys)-1] == v1.TopologyKeyAny {
			// edge case: include all endpoints if topology key "Any" specified
			// when we cannot determine current node's topology.
			return endpoints
		}
		// edge case: do not include any endpoints if topology key "Any" is
		// not specified when we cannot determine current node's topology.
		return filteredEndpoint
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
				filteredEndpoint = append(filteredEndpoint, ep)
			}
		}
		if len(filteredEndpoint) > 0 {
			return filteredEndpoint
		}
	}
	return filteredEndpoint
}
