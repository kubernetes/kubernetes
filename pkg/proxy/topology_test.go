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
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

func checkExpectedEndpoints(expected sets.Set[string], actual []Endpoint) error {
	var errs []error

	expectedCopy := sets.New[string](expected.UnsortedList()...)
	for _, ep := range actual {
		if !expectedCopy.Has(ep.String()) {
			errs = append(errs, fmt.Errorf("unexpected endpoint %v", ep))
		}
		expectedCopy.Delete(ep.String())
	}
	if len(expectedCopy) > 0 {
		errs = append(errs, fmt.Errorf("missing endpoints %v", expectedCopy.UnsortedList()))
	}

	return kerrors.NewAggregate(errs)
}

func TestCategorizeEndpoints(t *testing.T) {
	testCases := []struct {
		name        string
		pteEnabled  bool
		nodeLabels  map[string]string
		serviceInfo ServicePort
		endpoints   []Endpoint

		// We distinguish `nil` ("service doesn't use this kind of endpoints") from
		// `sets.Set[string]()` ("service uses this kind of endpoints but has no endpoints").
		// allEndpoints can be left unset if only one of clusterEndpoints and
		// localEndpoints is set, and allEndpoints is identical to it.
		// onlyRemoteEndpoints should be true if CategorizeEndpoints returns true for
		// hasAnyEndpoints despite allEndpoints being empty.
		clusterEndpoints    sets.Set[string]
		localEndpoints      sets.Set[string]
		allEndpoints        sets.Set[string]
		onlyRemoteEndpoints bool
	}{{
		name:        "should use topology since all endpoints have hints, node has a zone label and and there are endpoints for the node's zone",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80", "10.1.2.6:80"),
		localEndpoints:   nil,
	}, {
		name:        "externalTrafficPolicy: Local, topology ignored for Local endpoints",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80", "10.1.2.6:80"),
		localEndpoints:   sets.New[string]("10.1.2.3:80", "10.1.2.4:80"),
		allEndpoints:     sets.New[string]("10.1.2.3:80", "10.1.2.4:80", "10.1.2.6:80"),
	}, {
		name:        "internalTrafficPolicy: Local, topology ignored for Local endpoints",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true, externalPolicyLocal: false, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80", "10.1.2.6:80"),
		localEndpoints:   sets.New[string]("10.1.2.3:80", "10.1.2.4:80"),
		allEndpoints:     sets.New[string]("10.1.2.3:80", "10.1.2.4:80", "10.1.2.6:80"),
	}, {
		name:        "empty node labels",
		nodeLabels:  map[string]string{},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80"),
		localEndpoints:   nil,
	}, {
		name:        "empty zone label",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: ""},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80"),
		localEndpoints:   nil,
	}, {
		name:        "node in different zone, no endpoint filtering",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-b"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80"),
		localEndpoints:   nil,
	}, {
		name:        "unready endpoint",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: false}, // unready
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80"),
		localEndpoints:   nil,
	}, {
		name:        "only unready endpoints in same zone (should not filter)",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: false},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: false},
		},
		clusterEndpoints: sets.New[string]("10.1.2.4:80", "10.1.2.5:80"),
		localEndpoints:   nil,
	}, {
		name:        "missing hints, no filtering applied",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: nil, ready: true}, // Endpoint is missing hint.
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-a"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
		localEndpoints:   nil,
	}, {
		name:        "multiple hints per endpoint, filtering includes any endpoint with zone included",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-c"},
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.1.2.3:80", zoneHints: sets.New[string]("zone-a", "zone-b", "zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.4:80", zoneHints: sets.New[string]("zone-b", "zone-c"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.5:80", zoneHints: sets.New[string]("zone-b", "zone-d"), ready: true},
			&BaseEndpointInfo{endpoint: "10.1.2.6:80", zoneHints: sets.New[string]("zone-c"), ready: true},
		},
		clusterEndpoints: sets.New[string]("10.1.2.3:80", "10.1.2.4:80", "10.1.2.6:80"),
		localEndpoints:   nil,
	}, {
		name:        "conflicting topology and localness require merging allEndpoints",
		nodeLabels:  map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: false, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", zoneHints: sets.New[string]("zone-a"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", zoneHints: sets.New[string]("zone-b"), ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.2:80", zoneHints: sets.New[string]("zone-a"), ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.3:80", zoneHints: sets.New[string]("zone-b"), ready: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.2:80"),
		localEndpoints:   sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80", "10.0.0.2:80"),
	}, {
		name:             "internalTrafficPolicy: Local, with empty endpoints",
		serviceInfo:      &BaseServicePortInfo{internalPolicyLocal: true},
		endpoints:        []Endpoint{},
		clusterEndpoints: nil,
		localEndpoints:   sets.New[string](),
	}, {
		name:        "internalTrafficPolicy: Local, but all endpoints are remote",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints:    nil,
		localEndpoints:      sets.New[string](),
		onlyRemoteEndpoints: true,
	}, {
		name:        "internalTrafficPolicy: Local, all endpoints are local",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: true},
		},
		clusterEndpoints: nil,
		localEndpoints:   sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "internalTrafficPolicy: Local, some endpoints are local",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints: nil,
		localEndpoints:   sets.New[string]("10.0.0.0:80"),
	}, {
		name:        "Cluster traffic policy, endpoints not Ready",
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: false},
		},
		clusterEndpoints: sets.New[string](),
		localEndpoints:   nil,
	}, {
		name:        "Cluster traffic policy, some endpoints are Ready",
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true},
		},
		clusterEndpoints: sets.New[string]("10.0.0.1:80"),
		localEndpoints:   nil,
	}, {
		name:        "Cluster traffic policy, all endpoints are terminating",
		pteEnabled:  true,
		serviceInfo: &BaseServicePortInfo{},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: false, serving: true, terminating: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: false, serving: true, terminating: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   nil,
	}, {
		name:        "iTP: Local, eTP: Cluster, some endpoints local",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true, externalPolicyLocal: false, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   sets.New[string]("10.0.0.0:80"),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "iTP: Cluster, eTP: Local, some endpoints local",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: false, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   sets.New[string]("10.0.0.0:80"),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "iTP: Local, eTP: Local, some endpoints local",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   sets.New[string]("10.0.0.0:80"),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "iTP: Local, eTP: Local, all endpoints remote",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   sets.New[string](),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "iTP: Local, eTP: Local, all endpoints remote and terminating",
		pteEnabled:  true,
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: false, serving: true, terminating: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: false, serving: true, terminating: true, isLocal: false},
		},
		clusterEndpoints:    sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:      sets.New[string](),
		allEndpoints:        sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		onlyRemoteEndpoints: true,
	}, {
		name:        "iTP: Cluster, eTP: Local, with terminating endpoints",
		pteEnabled:  true,
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: false, externalPolicyLocal: true, nodePort: 8080},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: false, serving: false, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.2:80", ready: false, serving: true, terminating: true, isLocal: true},
			&BaseEndpointInfo{endpoint: "10.0.0.3:80", ready: false, serving: true, terminating: true, isLocal: false},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80"),
		localEndpoints:   sets.New[string]("10.0.0.2:80"),
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.2:80"),
	}, {
		name:        "externalTrafficPolicy ignored if not externally accessible",
		serviceInfo: &BaseServicePortInfo{externalPolicyLocal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: true},
		},
		clusterEndpoints: sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
		localEndpoints:   nil,
		allEndpoints:     sets.New[string]("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "no cluster endpoints for iTP:Local internal-only service",
		serviceInfo: &BaseServicePortInfo{internalPolicyLocal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{endpoint: "10.0.0.0:80", ready: true, isLocal: false},
			&BaseEndpointInfo{endpoint: "10.0.0.1:80", ready: true, isLocal: true},
		},
		clusterEndpoints: nil,
		localEndpoints:   sets.New[string]("10.0.0.1:80"),
		allEndpoints:     sets.New[string]("10.0.0.1:80"),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clusterEndpoints, localEndpoints, allEndpoints, hasAnyEndpoints := CategorizeEndpoints(tc.endpoints, tc.serviceInfo, tc.nodeLabels)

			if tc.clusterEndpoints == nil && clusterEndpoints != nil {
				t.Errorf("expected no cluster endpoints but got %v", clusterEndpoints)
			} else {
				err := checkExpectedEndpoints(tc.clusterEndpoints, clusterEndpoints)
				if err != nil {
					t.Errorf("error with cluster endpoints: %v", err)
				}
			}

			if tc.localEndpoints == nil && localEndpoints != nil {
				t.Errorf("expected no local endpoints but got %v", localEndpoints)
			} else {
				err := checkExpectedEndpoints(tc.localEndpoints, localEndpoints)
				if err != nil {
					t.Errorf("error with local endpoints: %v", err)
				}
			}

			var expectedAllEndpoints sets.Set[string]
			if tc.clusterEndpoints != nil && tc.localEndpoints == nil {
				expectedAllEndpoints = tc.clusterEndpoints
			} else if tc.localEndpoints != nil && tc.clusterEndpoints == nil {
				expectedAllEndpoints = tc.localEndpoints
			} else {
				expectedAllEndpoints = tc.allEndpoints
			}
			err := checkExpectedEndpoints(expectedAllEndpoints, allEndpoints)
			if err != nil {
				t.Errorf("error with allEndpoints: %v", err)
			}

			expectedHasAnyEndpoints := len(expectedAllEndpoints) > 0 || tc.onlyRemoteEndpoints
			if expectedHasAnyEndpoints != hasAnyEndpoints {
				t.Errorf("expected hasAnyEndpoints=%v, got %v", expectedHasAnyEndpoints, hasAnyEndpoints)
			}
		})
	}
}
