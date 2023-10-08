/*
Copyright 2023 The Kubernetes Authors.

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

package topologyheuristics

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestPreferZoneHeuristic_PopulateHints(t *testing.T) {

	testCases := []struct {
		name      string
		sliceInfo *SliceInfo

		wantSlicesToCreate []*discoveryv1.EndpointSlice
		wantSlicesToUpdate []*discoveryv1.EndpointSlice
		wantEvents         []*EventBuilder
	}{
		{
			name: "normal",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				ToCreate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.1"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
							{
								Addresses:  []string{"10.0.0.2"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.3"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
							{
								Addresses:  []string{"10.0.0.4"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
				},
				ToUpdate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.5"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
							{
								Addresses:  []string{"10.0.0.6"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.7"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
							{
								Addresses:  []string{"10.0.0.8"},
								Zone:       ptr.To("zone-c"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
				},
			},
			wantSlicesToCreate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.1"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
						{
							Addresses:  []string{"10.0.0.2"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
					},
				},
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.3"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
						{
							Addresses:  []string{"10.0.0.4"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
					},
				},
			},
			wantSlicesToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.5"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
						{
							Addresses:  []string{"10.0.0.6"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
					},
				},
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.7"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
						{
							Addresses:  []string{"10.0.0.8"},
							Zone:       ptr.To("zone-c"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-c"}}},
						},
					},
				},
			},
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			}},
		},
		{
			name: "endpoints with missing zone should not set any hints and remove already present hints",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				ToCreate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								// This endpoint does have the Zone set.
								Addresses:  []string{"10.0.0.1"},
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
							{
								Addresses:  []string{"10.0.0.2"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
				},
				ToUpdate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.3"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								// Already set hints should get removed.
								Hints: &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
							},
						},
					},
				},
			},
			wantSlicesToCreate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.1"},
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
						},
						{
							Addresses:  []string{"10.0.0.2"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
						},
					},
				},
			},
			wantSlicesToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.3"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
						},
					},
				},
			},
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   NoZoneSpecified,
			}},
		},
		{
			name: "unready endpoint with missing zone should be ignored",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				ToUpdate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								// This endpoint has missing zone but is not ready so should be ignored.
								Addresses:  []string{"10.0.0.1"},
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(false)},
							},
							{
								Addresses:  []string{"10.0.0.2"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
				},
			},
			wantSlicesToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.1"},
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(false)},
						},
						{
							Addresses:  []string{"10.0.0.2"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
					},
				},
			},
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			}},
		},
		{
			name: "unchanged endpoint slices need update because zone hint is missing",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				ToUpdate: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.1"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							},
						},
					},
				},
				Unchanged: []*discoveryv1.EndpointSlice{
					{
						// Endpoint slice needs update.
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.2"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
							},
							{
								Addresses:  []string{"10.0.0.3"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								// Zone hint missing.
							},
						},
					},
					{
						// Endpoint slice does NOT need update.
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.4"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
							},
						},
					},
					{
						// Endpoint slice needs update
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.5"},
								Zone:       ptr.To("zone-a"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								// Zone hint missing.
							},
						},
					},
				},
			},
			wantSlicesToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.1"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
					},
				},
				{
					// Endpoint slice moved from unchanged.
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.2"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
						{
							Addresses:  []string{"10.0.0.3"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
					},
				},
				{
					// Endpoint slice moved from unchanged.
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.5"},
							Zone:       ptr.To("zone-a"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
						},
					},
				},
			},
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			}},
		},
		{
			name: "unchanged endpoint slices need update because zone hint is incorrect",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				Unchanged: []*discoveryv1.EndpointSlice{
					{
						// Endpoint slice needs update because zone hint is incorrect.
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.1"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-a"}}},
							},
						},
					},
				},
			},
			wantSlicesToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses:  []string{"10.0.0.1"},
							Zone:       ptr.To("zone-b"),
							Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
							Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
						},
					},
				},
			},
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			}},
		},
		{
			name: "unchanged endpoints need no update",
			sliceInfo: &SliceInfo{
				ServiceKey:  "ns/svc",
				AddressType: discoveryv1.AddressTypeIPv4,
				Unchanged: []*discoveryv1.EndpointSlice{
					{
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses:  []string{"10.0.0.1"},
								Zone:       ptr.To("zone-b"),
								Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
								Hints:      &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "zone-b"}}},
							},
						},
					},
				},
			},
			wantSlicesToCreate: nil,
			wantSlicesToUpdate: nil,
			wantEvents: []*EventBuilder{{
				EventType: corev1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			heuristic := NewPreferZoneHeuristic()
			logger, _ := ktesting.NewTestContext(t)

			gotSlicesToCreate, gotSlicesToUpdate, gotEvents := heuristic.PopulateHints(logger, tc.sliceInfo)

			if diff := cmp.Diff(tc.wantSlicesToCreate, gotSlicesToCreate); diff != "" {
				t.Errorf("PopulateHints(...) returned unexpected diff in 'slicesToCreate': (-want, +got)\n%v", diff)
			}
			if diff := cmp.Diff(tc.wantSlicesToUpdate, gotSlicesToUpdate); diff != "" {
				t.Errorf("PopulateHints(...) returned unexpected diff in 'slicesToUpdate': (-want, +got)\n%v", diff)
			}

			eventsComparer := cmp.Comparer(func(e1, e2 EventBuilder) bool {
				if e1.EventType != e2.EventType || e1.Reason != e2.Reason {
					return false
				}
				if !strings.Contains(e1.Message, e2.Message) && !strings.Contains(e2.Message, e1.Message) {
					return false
				}
				return true
			})
			if diff := cmp.Diff(tc.wantEvents, gotEvents, eventsComparer); diff != "" {
				t.Errorf("PopulateHints(...) returned unexpected diff in 'events': (-want, +got)\n%v", diff)
			}
		})
	}
}
