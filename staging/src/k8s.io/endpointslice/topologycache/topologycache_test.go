/*
Copyright 2021 The Kubernetes Authors.

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

package topologycache

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/pointer"
)

func TestAddHints(t *testing.T) {
	testCases := []struct {
		name                        string
		cpuRatiosByZone             map[string]float64
		sliceInfo                   *SliceInfo
		expectedEndpointsByAddrType map[discovery.AddressType]EndpointZoneInfo
		expectedSlicesToCreate      []*discovery.EndpointSlice
		expectedSlicesToUpdate      []*discovery.EndpointSlice
		expectedEvents              []*EventBuilder
	}{{
		name:            "empty",
		cpuRatiosByZone: nil,
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
		},
		expectedEndpointsByAddrType: nil,
		expectedSlicesToCreate:      []*discovery.EndpointSlice{},
		expectedSlicesToUpdate:      []*discovery.EndpointSlice{},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   InsufficientNodeInfo,
			},
		},
	}, {
		name:            "slice to create, no zone ratios",
		cpuRatiosByZone: nil,
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
			ToCreate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}},
		},
		expectedEndpointsByAddrType: nil,
		expectedSlicesToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		expectedSlicesToUpdate: []*discovery.EndpointSlice{},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   InsufficientNodeInfo,
			},
		},
	}, {
		name: "slice to create with 2 endpoints, zone ratios require 3",
		cpuRatiosByZone: map[string]float64{
			"zone-a": 0.3,
			"zone-b": 0.4,
			"zone-c": 0.3,
		},
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
			ToCreate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.2.4"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}},
		},
		expectedEndpointsByAddrType: nil,
		expectedSlicesToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.4"},
				Zone:       pointer.String("zone-b"),
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		expectedSlicesToUpdate: []*discovery.EndpointSlice{},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   InsufficientNumberOfEndpoints,
			},
		},
	}, {
		name: "slice to create with 2 endpoints, zone ratios only require 2",
		cpuRatiosByZone: map[string]float64{
			"zone-a": 0.45,
			"zone-b": 0.55,
		},
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
			ToCreate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.2.4"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}},
		},
		expectedEndpointsByAddrType: map[discovery.AddressType]EndpointZoneInfo{
			discovery.AddressTypeIPv4: {
				"zone-a": 1,
				"zone-b": 1,
			},
		},
		expectedSlicesToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.4"},
				Zone:       pointer.String("zone-b"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		expectedSlicesToUpdate: []*discovery.EndpointSlice{},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			},
		},
	}, {
		name: "slice to create with 2 ready, 1 unready, 1 unknown endpoints, zone ratios only require 2",
		cpuRatiosByZone: map[string]float64{
			"zone-a": 0.45,
			"zone-b": 0.55,
		},
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
			ToCreate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.2.4"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.2.5"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
				}, {
					Addresses: []string{"10.1.2.6"},
					Zone:      pointer.String("zone-b"),
				}},
			}},
		},
		expectedEndpointsByAddrType: map[discovery.AddressType]EndpointZoneInfo{
			discovery.AddressTypeIPv4: {
				"zone-a": 1,
				"zone-b": 1,
			},
		},
		expectedSlicesToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.4"},
				Zone:       pointer.String("zone-b"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.5"},
				Zone:       pointer.String("zone-b"),
				Hints:      nil,
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
			}, {
				Addresses: []string{"10.1.2.6"},
				Zone:      pointer.String("zone-b"),
				Hints:     nil,
			}},
		}},
		expectedSlicesToUpdate: []*discovery.EndpointSlice{},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			},
		},
	}, {
		name: "slices to create and update within 3 zone threshold",
		cpuRatiosByZone: map[string]float64{
			"zone-a": 0.35,
			"zone-b": 0.35,
			"zone-c": 0.30,
		},
		sliceInfo: &SliceInfo{
			ServiceKey:  "ns/svc",
			AddressType: discovery.AddressTypeIPv4,
			ToCreate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.2.4"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}, {
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.1.3.3"},
					Zone:       pointer.String("zone-c"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.3.4"},
					Zone:       pointer.String("zone-c"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.1.3.4"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}},
			ToUpdate: []*discovery.EndpointSlice{{
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.2.2.3"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.2.2.4"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}, {
				Endpoints: []discovery.Endpoint{{
					Addresses:  []string{"10.2.3.3"},
					Zone:       pointer.String("zone-b"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.2.3.4"},
					Zone:       pointer.String("zone-c"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}, {
					Addresses:  []string{"10.2.3.4"},
					Zone:       pointer.String("zone-a"),
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				}},
			}},
		},
		expectedEndpointsByAddrType: map[discovery.AddressType]EndpointZoneInfo{
			discovery.AddressTypeIPv4: {
				"zone-a": 4,
				"zone-b": 3,
				"zone-c": 3,
			},
		},
		expectedSlicesToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.4"},
				Zone:       pointer.String("zone-b"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}, {
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.3.3"},
				Zone:       pointer.String("zone-c"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-c"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.3.4"},
				Zone:       pointer.String("zone-c"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-c"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.3.4"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		expectedSlicesToUpdate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.2.2.3"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.2.2.4"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}, {
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.2.3.3"},
				Zone:       pointer.String("zone-b"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.2.3.4"},
				Zone:       pointer.String("zone-c"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-c"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.2.3.4"},
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		expectedEvents: []*EventBuilder{
			{
				EventType: v1.EventTypeNormal,
				Reason:    "TopologyAwareHintsEnabled",
				Message:   TopologyAwareHintsEnabled,
			},
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cache := NewTopologyCache()
			cache.cpuRatiosByZone = tc.cpuRatiosByZone

			logger, _ := ktesting.NewTestContext(t)
			slicesToCreate, slicesToUpdate, events := cache.AddHints(logger, tc.sliceInfo)

			expectEquivalentSlices(t, slicesToCreate, tc.expectedSlicesToCreate)
			expectEquivalentSlices(t, slicesToUpdate, tc.expectedSlicesToUpdate)
			compareExpectedEvents(t, tc.expectedEvents, events)

			endpointsByAddrType, ok := cache.endpointsByService[tc.sliceInfo.ServiceKey]
			if tc.expectedEndpointsByAddrType == nil {
				if ok {
					t.Errorf("Expected no endpoints for Service %s, got %+v", tc.sliceInfo.ServiceKey, endpointsByAddrType)
				}
			} else {
				if len(tc.expectedEndpointsByAddrType) != len(endpointsByAddrType) {
					t.Fatalf("Expected endpoints for %d address types, got %d", len(tc.expectedEndpointsByAddrType), len(endpointsByAddrType))
				}
				for addrType, expectedEndpointZoneInfo := range tc.expectedEndpointsByAddrType {
					endpointZoneInfo, ok := endpointsByAddrType[addrType]
					if !ok {
						t.Fatalf("Expected endpoints for %s address type, got none", addrType)
					}

					if len(expectedEndpointZoneInfo) != len(endpointZoneInfo) {
						t.Fatalf("Expected endpoints for %d zones, got %d", len(expectedEndpointZoneInfo), len(endpointZoneInfo))
					}

					for zone, expectedNum := range expectedEndpointZoneInfo {
						num, ok := endpointZoneInfo[zone]
						if !ok {
							t.Fatalf("Expected endpoints for %s zone, got none", zone)
						}
						if num != expectedNum {
							t.Errorf("Expected %d endpoints for %s zone, got %d", expectedNum, zone, num)
						}
					}
				}
			}
		})
	}
}

func TestSetNodes(t *testing.T) {
	type nodeInfo struct {
		zone   string
		cpu    resource.Quantity
		ready  v1.ConditionStatus
		labels map[string]string
	}

	testCases := []struct {
		name                     string
		nodes                    []nodeInfo
		expectSufficientNodeInfo bool
		expectedCPUByZone        map[string]*resource.Quantity
		expectedRatios           map[string]float64
	}{{
		name:                     "empty",
		nodes:                    []nodeInfo{},
		expectSufficientNodeInfo: false,
		expectedCPUByZone:        nil,
		expectedRatios:           nil,
	}, {
		name: "single node",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		},
		expectSufficientNodeInfo: false,
		expectedCPUByZone:        nil,
		expectedRatios:           nil,
	}, {
		name: "single zone",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		},
		expectSufficientNodeInfo: false,
		expectedCPUByZone:        nil,
		expectedRatios:           nil,
	}, {
		name: "2 zones",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		},
		expectSufficientNodeInfo: true,
		expectedCPUByZone: map[string]*resource.Quantity{
			"zone-a": resource.NewQuantity(1, resource.BinarySI),
			"zone-b": resource.NewQuantity(1, resource.BinarySI),
		},
		expectedRatios: map[string]float64{
			"zone-a": 0.5,
			"zone-b": 0.5,
		},
	}, {
		name: "2 zones, unready node in 1, ready node in 1",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionFalse},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		},
		expectSufficientNodeInfo: false,
		expectedCPUByZone:        nil,
		expectedRatios:           nil,
	}, {
		name: "2 zones, control plane node in 1, ready node in 1",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue,
				labels: map[string]string{"node-role.kubernetes.io/control-plane": ""}},
		},
		expectSufficientNodeInfo: false,
		expectedCPUByZone:        nil,
		expectedRatios:           nil,
	}, {
		name: "2 zones, unready node in 1, ready node in 2",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionFalse},
		},
		expectSufficientNodeInfo: true,
		expectedCPUByZone: map[string]*resource.Quantity{
			"zone-a": resource.NewQuantity(1, resource.BinarySI),
			"zone-b": resource.NewQuantity(1, resource.BinarySI),
		},
		expectedRatios: map[string]float64{
			"zone-a": 0.5,
			"zone-b": 0.5,
		},
	}, {
		name: "2 zones, control plane node in 1, ready node in 2",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue,
				labels: map[string]string{"node-role.kubernetes.io/control-plane": ""}},
		},
		expectSufficientNodeInfo: true,
		expectedCPUByZone: map[string]*resource.Quantity{
			"zone-a": resource.NewQuantity(1, resource.BinarySI),
			"zone-b": resource.NewQuantity(1, resource.BinarySI),
		},
		expectedRatios: map[string]float64{
			"zone-a": 0.5,
			"zone-b": 0.5,
		},
	}, {
		name: "3 zones, 4 nodes in 1, 2 nodes in 1, 1 node in 1",
		nodes: []nodeInfo{
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
			{zone: "zone-a", cpu: resource.MustParse("2000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("3000m"), ready: v1.ConditionTrue},
			{zone: "zone-b", cpu: resource.MustParse("1500m"), ready: v1.ConditionTrue},
			{zone: "zone-c", cpu: resource.MustParse("500m"), ready: v1.ConditionTrue},
		},
		expectSufficientNodeInfo: true,
		expectedCPUByZone: map[string]*resource.Quantity{
			"zone-a": resource.NewMilliQuantity(5000, resource.BinarySI),
			"zone-b": resource.NewMilliQuantity(4500, resource.BinarySI),
			"zone-c": resource.NewMilliQuantity(500, resource.BinarySI),
		},
		expectedRatios: map[string]float64{
			"zone-a": 0.5,
			"zone-b": 0.45,
			"zone-c": 0.05,
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cache := NewTopologyCache()
			nodes := make([]*v1.Node, 0, len(tc.nodes))
			for _, node := range tc.nodes {
				labels := node.labels
				if labels == nil {
					labels = map[string]string{}
				}
				if node.zone != "" {
					labels[v1.LabelTopologyZone] = node.zone
				}
				conditions := []v1.NodeCondition{{
					Type:   v1.NodeReady,
					Status: node.ready,
				}}
				allocatable := v1.ResourceList{
					v1.ResourceCPU: node.cpu,
				}
				nodes = append(nodes, &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Labels: labels,
					},
					Status: v1.NodeStatus{
						Allocatable: allocatable,
						Conditions:  conditions,
					},
				})
			}

			logger, _ := ktesting.NewTestContext(t)
			cache.SetNodes(logger, nodes)

			if cache.sufficientNodeInfo != tc.expectSufficientNodeInfo {
				t.Errorf("Expected sufficientNodeInfo to be %t, got %t", tc.expectSufficientNodeInfo, cache.sufficientNodeInfo)
			}

			if cache.cpuRatiosByZone == nil || tc.expectedRatios == nil {
				if (cache.cpuRatiosByZone == nil) != (tc.expectedRatios == nil) {
					t.Errorf("Expected %+v, got %+v", tc.expectedRatios, cache.cpuRatiosByZone)
				}
			} else {
				if len(cache.cpuRatiosByZone) != len(tc.expectedRatios) {
					t.Errorf("Expected ratios with %d zones, got %d", len(tc.expectedRatios), len(cache.cpuRatiosByZone))
				}
				for zone, expectedRatio := range tc.expectedRatios {
					actualRatio, ok := cache.cpuRatiosByZone[zone]
					if !ok {
						t.Errorf("Expected ratio for %s zone, got none", zone)
					} else if actualRatio != expectedRatio {
						t.Errorf("Expected ratio to be %f, got %f", expectedRatio, actualRatio)
					}
				}
			}

			if cache.cpuByZone == nil || tc.expectedCPUByZone == nil {
				if (cache.cpuByZone == nil) != (tc.expectedCPUByZone == nil) {
					t.Errorf("Expected %+v, got %+v", tc.expectedCPUByZone, cache.cpuByZone)
				}
			} else {
				if len(cache.cpuByZone) != len(tc.expectedCPUByZone) {
					t.Errorf("Expected CPU with %d zones, got %d", len(tc.expectedCPUByZone), len(cache.cpuByZone))
				}
				for zone, expectedCPU := range tc.expectedCPUByZone {
					actualCPU, ok := cache.cpuByZone[zone]
					if !ok {
						t.Errorf("Expected CPU for %s zone, got none", zone)
					} else if !actualCPU.Equal(*expectedCPU) {
						t.Errorf("Expected CPU to be %d, got %d", expectedCPU.MilliValue(), actualCPU.MilliValue())
					}
				}
			}
		})
	}
}

func TestTopologyCacheRace(t *testing.T) {
	sliceInfo := &SliceInfo{
		ServiceKey:  "ns/svc",
		AddressType: discovery.AddressTypeIPv4,
		ToCreate: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Addresses:  []string{"10.1.2.3"},
				Zone:       pointer.String("zone-a"),
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Addresses:  []string{"10.1.2.4"},
				Zone:       pointer.String("zone-b"),
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}}}
	type nodeInfo struct {
		zone  string
		cpu   resource.Quantity
		ready v1.ConditionStatus
	}
	nodeInfos := []nodeInfo{
		{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		{zone: "zone-a", cpu: resource.MustParse("1000m"), ready: v1.ConditionTrue},
		{zone: "zone-a", cpu: resource.MustParse("2000m"), ready: v1.ConditionTrue},
		{zone: "zone-b", cpu: resource.MustParse("3000m"), ready: v1.ConditionTrue},
		{zone: "zone-b", cpu: resource.MustParse("1500m"), ready: v1.ConditionTrue},
		{zone: "zone-c", cpu: resource.MustParse("500m"), ready: v1.ConditionTrue},
	}

	cache := NewTopologyCache()
	nodes := []*v1.Node{}
	for _, node := range nodeInfos {
		labels := map[string]string{}
		if node.zone != "" {
			labels[v1.LabelTopologyZone] = node.zone
		}
		conditions := []v1.NodeCondition{{
			Type:   v1.NodeReady,
			Status: node.ready,
		}}
		allocatable := v1.ResourceList{
			v1.ResourceCPU: node.cpu,
		}
		nodes = append(nodes, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: labels,
			},
			Status: v1.NodeStatus{
				Allocatable: allocatable,
				Conditions:  conditions,
			},
		})
	}

	logger, _ := ktesting.NewTestContext(t)
	go func() {
		cache.SetNodes(logger, nodes)
	}()
	go func() {
		cache.AddHints(logger, sliceInfo)
	}()
	go func() {
		cache.HasPopulatedHints(sliceInfo.ServiceKey)
	}()
}

// Test Helpers

func expectEquivalentSlices(t *testing.T, actualSlices, expectedSlices []*discovery.EndpointSlice) {
	t.Helper()

	if len(actualSlices) != len(expectedSlices) {
		t.Fatalf("Expected %d slices, got %d", len(expectedSlices), len(actualSlices))
	}

	for i, expectedSlice := range expectedSlices {
		actualSlice := actualSlices[i]

		if len(expectedSlice.Endpoints) != len(actualSlice.Endpoints) {
			t.Errorf("Expected %d endpoints, got %d", len(expectedSlice.Endpoints), len(actualSlice.Endpoints))
			continue
		}
		for j, expectedEndpoint := range expectedSlice.Endpoints {
			actualEndpoint := actualSlice.Endpoints[j]
			if !reflect.DeepEqual(actualEndpoint, expectedEndpoint) {
				t.Errorf("Endpoints didn't match\nExpected: %+v\nGot: %+v", expectedEndpoint, actualEndpoint)
			}
		}
	}
}

func compareExpectedEvents(t *testing.T, expectedEvents, events []*EventBuilder) {
	if len(expectedEvents) != len(events) {
		t.Errorf("Expected %d event, got %d", len(expectedEvents), len(events))
	}
	for i, event := range events {
		if diff := cmp.Diff(event, expectedEvents[i], cmpopts.IgnoreFields(EventBuilder{}, "Message")); diff != "" {
			t.Errorf("Unexpected event (-want,+got):\n%s", diff)
		}
		if got, want := event.Message, expectedEvents[i].Message; !strings.HasPrefix(got, want) || want == "" {
			t.Errorf("Unexpected event message:\ngot %q want a message with %q prefix", got, want)
		}
	}
}
