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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestFilterTopologyEndpoint(t *testing.T) {
	type endpoint struct {
		Endpoint string
		NodeName types.NodeName
	}
	testCases := []struct {
		Name            string
		nodeLabels      map[types.NodeName]map[string]string
		endpoints       []endpoint
		currentNodeName types.NodeName
		topologyKeys    []string
		expected        []endpoint
	}{
		{
			// Case[0]: no topology key and endpoints at all = 0 endpoints
			Name: "no topology key and endpoints",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "10.0.0.1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				}},
			endpoints:       []endpoint{},
			currentNodeName: "testNode1",
			topologyKeys:    nil,
			expected:        []endpoint{},
		},
		{
			// Case[1]: no topology key, 2 nodes each with 2 endpoints = 4
			// endpoints
			Name: "no topology key but have endpoints",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
			currentNodeName: "testNode1",
			topologyKeys:    nil,
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
		},
		{
			// Case[2]: 1 topology key (hostname), 2 nodes each with 2 endpoints
			// 1 match = 2 endpoints
			Name: "one topology key with one node matched",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
			currentNodeName: "testNode1",
			topologyKeys:    []string{"kubernetes.io/hostname"},
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
			},
		},
		{
			// Case[3]: 1 topology key (hostname), 2 nodes each with 2 endpoints
			// no match = 0 endpoints
			Name: "one topology key without node matched",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
			currentNodeName: "testNode3",
			topologyKeys:    []string{"kubernetes.io/hostname"},
			expected:        []endpoint{},
		},
		{
			// Case[4]: 1 topology key (zone), 2 nodes in zone a, 2 nodes in
			// zone b, each with 2 endpoints = 4 endpoints
			Name: "one topology key with multiple nodes matched",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode4": {
					"kubernetes.io/hostname":        "testNode4",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
			currentNodeName: "testNode2",
			topologyKeys:    []string{"topology.kubernetes.io/zone"},
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
		},
		{
			// Case[5]: 2 topology keys (hostname, zone), 2 nodes each with 2
			// endpoints, 1 hostname match = 2 endpoints (2nd key ignored)
			Name: "early match in multiple topology keys",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode4": {
					"kubernetes.io/hostname":        "testNode4",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
			currentNodeName: "testNode2",
			topologyKeys:    []string{"kubernetes.io/hostname"},
			expected: []endpoint{
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
		},
		{
			// Case[6]: 2 topology keys (hostname, zone), 2 nodes in zone a, 2
			// nodes in zone b, each with 2 endpoints, no hostname match, 1 zone
			// match = 4 endpoints
			Name: "later match in multiple topology keys",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode4": {
					"kubernetes.io/hostname":        "testNode4",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode5": {
					"kubernetes.io/hostname":        "testNode5",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
			currentNodeName: "testNode5",
			topologyKeys:    []string{"kubernetes.io/hostname", "topology.kubernetes.io/zone"},
			expected: []endpoint{
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
		},
		{
			// Case[7]: 2 topology keys (hostname, zone), 2 nodes in zone a, 2
			// nodes in zone b, each with 2 endpoints, no hostname match, no zone
			// match = 0 endpoints
			Name: "multiple topology keys without node matched",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode4": {
					"kubernetes.io/hostname":        "testNode4",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode5": {
					"kubernetes.io/hostname":        "testNode5",
					"topology.kubernetes.io/zone":   "90003",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
			currentNodeName: "testNode5",
			topologyKeys:    []string{"kubernetes.io/hostname", "topology.kubernetes.io/zone"},
			expected:        []endpoint{},
		},
		{
			// Case[8]: 2 topology keys (hostname, "*"), 2 nodes each with 2
			// endpoints, 1 match hostname = 2 endpoints
			Name: "multiple topology keys matched node when 'Any' key ignored",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
			currentNodeName: "testNode1",
			topologyKeys:    []string{"kubernetes.io/hostname", v1.TopologyKeyAny},
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
			},
		},
		{
			// Case[9]: 2 topology keys (hostname, "*"), 2 nodes each with 2
			// endpoints, no hostname match, catch-all ("*") matched with 4
			// endpoints
			Name: "two topology keys matched node with 'Any' key",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
			currentNodeName: "testNode3",
			topologyKeys:    []string{"kubernetes.io/hostname", v1.TopologyKeyAny},
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
			},
		},
		{
			// Case[10]: 3 topology keys (hostname, zone, "*"), 2 nodes in zone a,
			// 2 nodes in zone b, each with 2 endpoints, no hostname match, no
			// zone, catch-all ("*") matched with 8 endpoints
			Name: "multiple topology keys matched node with 'Any' key",
			nodeLabels: map[types.NodeName]map[string]string{
				"testNode1": {
					"kubernetes.io/hostname":        "testNode1",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode2": {
					"kubernetes.io/hostname":        "testNode2",
					"topology.kubernetes.io/zone":   "90001",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode3": {
					"kubernetes.io/hostname":        "testNode3",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode4": {
					"kubernetes.io/hostname":        "testNode4",
					"topology.kubernetes.io/zone":   "90002",
					"topology.kubernetes.io/region": "cd",
				},
				"testNode5": {
					"kubernetes.io/hostname":        "testNode5",
					"topology.kubernetes.io/zone":   "90003",
					"topology.kubernetes.io/region": "cd",
				},
			},
			endpoints: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
			currentNodeName: "testNode5",
			topologyKeys:    []string{"kubernetes.io/hostname", "topology.kubernetes.io/zone", v1.TopologyKeyAny},
			expected: []endpoint{
				{Endpoint: "1.1.1.1:11", NodeName: "testNode1"},
				{Endpoint: "1.1.1.2:11", NodeName: "testNode1"},
				{Endpoint: "1.1.2.1:11", NodeName: "testNode2"},
				{Endpoint: "1.1.2.2:11", NodeName: "testNode2"},
				{Endpoint: "1.1.3.1:11", NodeName: "testNode3"},
				{Endpoint: "1.1.3.2:11", NodeName: "testNode3"},
				{Endpoint: "1.1.4.1:11", NodeName: "testNode4"},
				{Endpoint: "1.1.4.2:11", NodeName: "testNode4"},
			},
		},
	}
	endpointsToStringArray := func(endpoints []endpoint) []string {
		result := make([]string, 0, len(endpoints))
		for _, ep := range endpoints {
			result = append(result, ep.Endpoint)
		}
		return result
	}
	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			m := make(map[Endpoint]endpoint)
			endpoints := []Endpoint{}
			for _, ep := range tc.endpoints {
				var e Endpoint = &BaseEndpointInfo{Endpoint: ep.Endpoint, Topology: tc.nodeLabels[ep.NodeName]}
				m[e] = ep
				endpoints = append(endpoints, e)
			}
			currentNodeLabels := tc.nodeLabels[tc.currentNodeName]
			filteredEndpoint := []endpoint{}
			for _, ep := range FilterTopologyEndpoint(currentNodeLabels, tc.topologyKeys, endpoints) {
				filteredEndpoint = append(filteredEndpoint, m[ep])
			}
			if !reflect.DeepEqual(filteredEndpoint, tc.expected) {
				t.Errorf("expected %v, got %v", endpointsToStringArray(tc.expected), endpointsToStringArray(filteredEndpoint))
			}
		})
	}
}
