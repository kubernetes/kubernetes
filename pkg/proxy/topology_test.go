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
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestFilterEndpoints(t *testing.T) {
	type endpoint struct {
		ip        string
		zoneHints sets.String
		unready   bool
	}
	testCases := []struct {
		name              string
		hintsEnabled      bool
		nodeLabels        map[string]string
		serviceInfo       ServicePort
		endpoints         []endpoint
		expectedEndpoints []endpoint
	}{{
		name:         "hints enabled, hints annotation == auto",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "auto"},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:         "hints, hints annotation == disabled, hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "disabled"},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:         "hints, hints annotation == aUto (wrong capitalization), hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "aUto"},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:         "hints, hints annotation empty, hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:         "node local endpoints, hints are ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: true},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}}

	endpointsToStringArray := func(endpoints []Endpoint) []string {
		result := make([]string, 0, len(endpoints))
		for _, ep := range endpoints {
			result = append(result, ep.String())
		}
		return result
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareHints, tc.hintsEnabled)()

			endpoints := []Endpoint{}
			for _, ep := range tc.endpoints {
				endpoints = append(endpoints, &BaseEndpointInfo{Endpoint: ep.ip, ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			expectedEndpoints := []Endpoint{}
			for _, ep := range tc.expectedEndpoints {
				expectedEndpoints = append(expectedEndpoints, &BaseEndpointInfo{Endpoint: ep.ip, ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			filteredEndpoints := FilterEndpoints(endpoints, tc.serviceInfo, tc.nodeLabels)
			if len(filteredEndpoints) != len(expectedEndpoints) {
				t.Errorf("expected %d filtered endpoints, got %d", len(expectedEndpoints), len(filteredEndpoints))
			}
			if !reflect.DeepEqual(filteredEndpoints, expectedEndpoints) {
				t.Errorf("expected %v, got %v", endpointsToStringArray(expectedEndpoints), endpointsToStringArray(filteredEndpoints))
			}
		})
	}
}

func Test_filterEndpointsWithHints(t *testing.T) {
	type endpoint struct {
		ip        string
		zoneHints sets.String
		unready   bool
	}
	testCases := []struct {
		name              string
		nodeLabels        map[string]string
		hintsAnnotation   string
		endpoints         []endpoint
		expectedEndpoints []endpoint
	}{{
		name:              "empty node labels",
		nodeLabels:        map[string]string{},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
	}, {
		name:              "empty zone label",
		nodeLabels:        map[string]string{v1.LabelTopologyZone: ""},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
	}, {
		name:              "node in different zone, no endpoint filtering",
		nodeLabels:        map[string]string{v1.LabelTopologyZone: "zone-b"},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
	}, {
		name:            "normal endpoint filtering, auto annotation",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "unready endpoint",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a"), unready: true},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "only unready endpoints in same zone (should not filter)",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a"), unready: true},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a"), unready: true},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a"), unready: true},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a"), unready: true},
		},
	}, {
		name:            "normal endpoint filtering, Auto annotation",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "Auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "hintsAnnotation empty, no filtering applied",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "hintsAnnotation disabled, no filtering applied",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "disabled",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "missing hints, no filtering applied",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-a"},
		hintsAnnotation: "auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5"},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5"},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
	}, {
		name:            "multiple hints per endpoint, filtering includes any endpoint with zone included",
		nodeLabels:      map[string]string{v1.LabelTopologyZone: "zone-c"},
		hintsAnnotation: "auto",
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a", "zone-b", "zone-c")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b", "zone-c")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-b", "zone-d")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-c")},
		},
		expectedEndpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a", "zone-b", "zone-c")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b", "zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-c")},
		},
	}}

	endpointsToStringArray := func(endpoints []Endpoint) []string {
		result := make([]string, 0, len(endpoints))
		for _, ep := range endpoints {
			result = append(result, ep.String())
		}
		return result
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			endpoints := []Endpoint{}
			for _, ep := range tc.endpoints {
				endpoints = append(endpoints, &BaseEndpointInfo{Endpoint: ep.ip, ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			expectedEndpoints := []Endpoint{}
			for _, ep := range tc.expectedEndpoints {
				expectedEndpoints = append(expectedEndpoints, &BaseEndpointInfo{Endpoint: ep.ip, ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			filteredEndpoints := filterEndpointsWithHints(endpoints, tc.hintsAnnotation, tc.nodeLabels)
			if len(filteredEndpoints) != len(expectedEndpoints) {
				t.Errorf("expected %d filtered endpoints, got %d", len(expectedEndpoints), len(filteredEndpoints))
			}
			if !reflect.DeepEqual(filteredEndpoints, expectedEndpoints) {
				t.Errorf("expected %v, got %v", endpointsToStringArray(expectedEndpoints), endpointsToStringArray(filteredEndpoints))
			}
		})
	}
}

func TestFilterLocalEndpoint(t *testing.T) {
	testCases := []struct {
		name      string
		endpoints []Endpoint
		expected  []Endpoint
	}{
		{
			name:      "with empty endpoints",
			endpoints: []Endpoint{},
			expected:  nil,
		},
		{
			name: "all endpoints not local",
			endpoints: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: false},
				&BaseEndpointInfo{Endpoint: "10.0.0.1:80", IsLocal: false},
			},
			expected: nil,
		},
		{
			name: "all endpoints are local",
			endpoints: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "10.0.0.1:80", IsLocal: true},
			},
			expected: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "10.0.0.1:80", IsLocal: true},
			},
		},
		{
			name: "some endpoints are local",
			endpoints: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "10.0.0.1:80", IsLocal: false},
			},
			expected: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: true},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			filteredEndpoint := FilterLocalEndpoint(tc.endpoints)
			if !reflect.DeepEqual(filteredEndpoint, tc.expected) {
				t.Errorf("expected %v, got %v", tc.expected, filteredEndpoint)
			}
		})
	}
}
