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
	testCases := []struct {
		name              string
		hintsEnabled      bool
		nodeLabels        map[string]string
		serviceInfo       ServicePort
		endpoints         []Endpoint
		expectedEndpoints sets.String
	}{{
		name:         "hints enabled, hints annotation == auto",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
	}, {
		name:         "hints, hints annotation == disabled, hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "disabled"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "hints, hints annotation == aUto (wrong capitalization), hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false, hintsAnnotation: "aUto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "hints, hints annotation empty, hints ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: false},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "node local endpoints, hints are ignored",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{nodeLocalExternal: true, hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "empty node labels",
		hintsEnabled: true,
		nodeLabels:   map[string]string{},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:         "empty zone label",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: ""},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:         "node in different zone, no endpoint filtering",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-b"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:         "normal endpoint filtering, auto annotation",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
	}, {
		name:         "unready endpoint",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: false},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80"),
	}, {
		name:         "only unready endpoints in same zone (should not filter)",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: false},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: false},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "normal endpoint filtering, Auto annotation",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "Auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.6:80"),
	}, {
		name:         "hintsAnnotation empty, no filtering applied",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: ""},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "hintsAnnotation disabled, no filtering applied",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "disabled"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "missing hints, no filtering applied",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-a"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: nil, Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-a"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.5:80", "10.1.2.6:80"),
	}, {
		name:         "multiple hints per endpoint, filtering includes any endpoint with zone included",
		hintsEnabled: true,
		nodeLabels:   map[string]string{v1.LabelTopologyZone: "zone-c"},
		serviceInfo:  &BaseServiceInfo{hintsAnnotation: "auto"},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.1.2.3:80", ZoneHints: sets.NewString("zone-a", "zone-b", "zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.4:80", ZoneHints: sets.NewString("zone-b", "zone-c"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.5:80", ZoneHints: sets.NewString("zone-b", "zone-d"), Ready: true},
			&BaseEndpointInfo{Endpoint: "10.1.2.6:80", ZoneHints: sets.NewString("zone-c"), Ready: true},
		},
		expectedEndpoints: sets.NewString("10.1.2.3:80", "10.1.2.4:80", "10.1.2.6:80"),
	}, {
		name:              "internalTrafficPolicy: Local, with empty endpoints",
		serviceInfo:       &BaseServiceInfo{nodeLocalInternal: true},
		endpoints:         []Endpoint{},
		expectedEndpoints: nil,
	}, {
		name:        "internalTrafficPolicy: Local, but all endpoints are remote",
		serviceInfo: &BaseServiceInfo{nodeLocalInternal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.0.0.0:80", Ready: true, IsLocal: false},
			&BaseEndpointInfo{Endpoint: "10.0.0.1:80", Ready: true, IsLocal: false},
		},
		expectedEndpoints: nil,
	}, {
		name:        "internalTrafficPolicy: Local, all endpoints are local",
		serviceInfo: &BaseServiceInfo{nodeLocalInternal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.0.0.0:80", Ready: true, IsLocal: true},
			&BaseEndpointInfo{Endpoint: "10.0.0.1:80", Ready: true, IsLocal: true},
		},
		expectedEndpoints: sets.NewString("10.0.0.0:80", "10.0.0.1:80"),
	}, {
		name:        "internalTrafficPolicy: Local, some endpoints are local",
		serviceInfo: &BaseServiceInfo{nodeLocalInternal: true},
		endpoints: []Endpoint{
			&BaseEndpointInfo{Endpoint: "10.0.0.0:80", Ready: true, IsLocal: true},
			&BaseEndpointInfo{Endpoint: "10.0.0.1:80", Ready: true, IsLocal: false},
		},
		expectedEndpoints: sets.NewString("10.0.0.0:80"),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareHints, tc.hintsEnabled)()

			filteredEndpoints := FilterEndpoints(tc.endpoints, tc.serviceInfo, tc.nodeLabels)
			err := checkExpectedEndpoints(tc.expectedEndpoints, filteredEndpoints)
			if err != nil {
				t.Errorf(err.Error())
			}
		})
	}
}
