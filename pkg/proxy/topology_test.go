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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func checkExpectedEndpoints(expected sets.String, actual []Endpoint) error {
	var errs []error

	expectedCopy := sets.NewString(expected.UnsortedList()...)
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
		expectedEndpoints sets.String
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "node local endpoints, hints are ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: true, hintsAnnotation: "auto"},
		endpoints: []endpoint{
			{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")},
			{ip: "10.1.2.4", zoneHints: sets.NewString("zone-b")},
			{ip: "10.1.2.5", zoneHints: sets.NewString("zone-c")},
			{ip: "10.1.2.6", zoneHints: sets.NewString("zone-a")},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareHints, tc.hintsEnabled)()

			endpoints := []Endpoint{}
			for _, ep := range tc.endpoints {
				endpoints = append(endpoints, &BaseEndpointInfo{Endpoint: ep.ip + ":80", ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			filteredEndpoints := FilterEndpoints(endpoints, tc.serviceInfo, tc.nodeLabels)
			err := checkExpectedEndpoints(tc.expectedEndpoints, filteredEndpoints)
			if err != nil {
				t.Errorf(err.Error())
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
		expectedEndpoints sets.String
	}{{
		name:              "empty node labels",
		nodeLabels:        map[string]string{},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:              "empty zone label",
		nodeLabels:        map[string]string{v1.LabelTopologyZone: ""},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:              "node in different zone, no endpoint filtering",
		nodeLabels:        map[string]string{v1.LabelTopologyZone: "zone-b"},
		hintsAnnotation:   "auto",
		endpoints:         []endpoint{{ip: "10.1.2.3", zoneHints: sets.NewString("zone-a")}},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
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
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.6:80"),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			endpoints := []Endpoint{}
			for _, ep := range tc.endpoints {
				endpoints = append(endpoints, &BaseEndpointInfo{Endpoint: ep.ip + ":80", ZoneHints: ep.zoneHints, Ready: !ep.unready})
			}

			filteredEndpoints := filterEndpointsWithHints(endpoints, tc.hintsAnnotation, tc.nodeLabels)
			err := checkExpectedEndpoints(tc.expectedEndpoints, filteredEndpoints)
			if err != nil {
				t.Errorf(err.Error())
			}
		})
	}
}

func TestFilterLocalEndpoint(t *testing.T) {
	testCases := []struct {
		name      string
		endpoints []Endpoint
		expected  sets.String
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
			expected: sets.NewString("10.0.0.0:80", "10.0.0.1:80"),
		},
		{
			name: "some endpoints are local",
			endpoints: []Endpoint{
				&BaseEndpointInfo{Endpoint: "10.0.0.0:80", IsLocal: true},
				&BaseEndpointInfo{Endpoint: "10.0.0.1:80", IsLocal: false},
			},
			expected: sets.NewString("10.0.0.0:80"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			filteredEndpoint := FilterLocalEndpoint(tc.endpoints)
			err := checkExpectedEndpoints(tc.expected, filteredEndpoint)
			if err != nil {
				t.Errorf(err.Error())
			}
		})
	}
}
