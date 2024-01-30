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
	"testing"

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/pointer"
)

func Test_redistributeHints(t *testing.T) {
	testCases := []struct {
		name                    string
		slices                  []*discovery.EndpointSlice
		givingZones             map[string]int
		receivingZones          map[string]int
		expectedRedistributions map[string]int
	}{{
		name:                    "empty",
		slices:                  []*discovery.EndpointSlice{},
		givingZones:             map[string]int{},
		receivingZones:          map[string]int{},
		expectedRedistributions: map[string]int{},
	}, {
		name: "single endpoint",
		slices: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		givingZones:             map[string]int{"zone-a": 1},
		receivingZones:          map[string]int{"zone-b": 1},
		expectedRedistributions: map[string]int{"zone-a": -1, "zone-b": 1},
	}, {
		name: "endpoints from 1 zone redistributed to 2 other zones",
		slices: []*discovery.EndpointSlice{{
			Endpoints: []discovery.Endpoint{{
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		}},
		givingZones:             map[string]int{"zone-a": 2},
		receivingZones:          map[string]int{"zone-b": 1, "zone-c": 1},
		expectedRedistributions: map[string]int{"zone-a": -2, "zone-b": 1, "zone-c": 1},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			actualRedistributions := redistributeHints(logger, tc.slices, tc.givingZones, tc.receivingZones)

			if len(actualRedistributions) != len(tc.expectedRedistributions) {
				t.Fatalf("Expected redistributions for %d zones, got %d (%+v)", len(tc.expectedRedistributions), len(actualRedistributions), actualRedistributions)
			}

			for zone, expectedNum := range tc.expectedRedistributions {
				actualNum, _ := actualRedistributions[zone]
				if actualNum != expectedNum {
					t.Errorf("Expected redistribution of %d for zone %s, got %d", expectedNum, zone, actualNum)
				}
			}
		})
	}
}

func Test_getGivingAndReceivingZones(t *testing.T) {
	testCases := []struct {
		name                   string
		allocations            map[string]allocation
		allocatedHintsByZone   map[string]int
		expectedGivingZones    map[string]int
		expectedReceivingZones map[string]int
	}{{
		name:                   "empty",
		allocations:            map[string]allocation{},
		allocatedHintsByZone:   map[string]int{},
		expectedGivingZones:    map[string]int{},
		expectedReceivingZones: map[string]int{},
	}, {
		name: "simple allocation with no need for rebalancing",
		allocations: map[string]allocation{
			"zone-a": {desired: 1.2},
			"zone-b": {desired: 1.1},
			"zone-c": {desired: 1.0},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": 1, "zone-b": 1, "zone-c": 1},
		expectedGivingZones:    map[string]int{},
		expectedReceivingZones: map[string]int{},
	}, {
		name: "preference for same zone even when giving an extra endpoint would result in slightly better distribution",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.1},
			"zone-b": {desired: 5.1},
			"zone-c": {desired: 5.8},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": 16},
		expectedGivingZones:    map[string]int{"zone-a": 10},
		expectedReceivingZones: map[string]int{"zone-b": 5, "zone-c": 5},
	}, {
		name: "when 2 zones need < 1 endpoint, give to zone that needs endpoint most",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.0},
			"zone-b": {desired: 5.6},
			"zone-c": {desired: 5.4},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": 16},
		expectedGivingZones:    map[string]int{"zone-a": 11},
		expectedReceivingZones: map[string]int{"zone-b": 6, "zone-c": 5},
	}, {
		name: "when 2 zones have extra endpoints, give from zone with most extra",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.0},
			"zone-b": {desired: 5.6},
			"zone-c": {desired: 5.4},
		},
		allocatedHintsByZone:   map[string]int{"zone-b": 8, "zone-c": 8},
		expectedGivingZones:    map[string]int{"zone-b": 2, "zone-c": 3},
		expectedReceivingZones: map[string]int{"zone-a": 5},
	}, {
		name: "ensure function can handle unexpected data (more allocated than allocations)",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.0},
			"zone-b": {desired: 5.0},
			"zone-c": {desired: 5.0},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": 6, "zone-b": 6, "zone-c": 6},
		expectedGivingZones:    map[string]int{},
		expectedReceivingZones: map[string]int{},
	}, {
		name: "ensure function can handle unexpected data (negative allocations)",
		allocations: map[string]allocation{
			"zone-a": {desired: -5.0},
			"zone-b": {desired: -5.0},
			"zone-c": {desired: -5.0},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": 6, "zone-b": 6, "zone-c": 6},
		expectedGivingZones:    map[string]int{},
		expectedReceivingZones: map[string]int{},
	}, {
		name: "ensure function can handle unexpected data (negative allocated)",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.0},
			"zone-b": {desired: 5.0},
			"zone-c": {desired: 5.0},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": -4, "zone-b": -3, "zone-c": -2},
		expectedGivingZones:    map[string]int{},
		expectedReceivingZones: map[string]int{},
	}, {
		name: "ensure function can handle unexpected data (negative for 1 zone)",
		allocations: map[string]allocation{
			"zone-a": {desired: 5.0},
			"zone-b": {desired: 5.0},
			"zone-c": {desired: 5.0},
		},
		allocatedHintsByZone:   map[string]int{"zone-a": -40, "zone-b": 20, "zone-c": 20},
		expectedGivingZones:    map[string]int{"zone-b": 15, "zone-c": 15},
		expectedReceivingZones: map[string]int{"zone-a": 30},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualGivingZones, actualReceivingZones := getGivingAndReceivingZones(tc.allocations, tc.allocatedHintsByZone)

			if !reflect.DeepEqual(actualGivingZones, tc.expectedGivingZones) {
				t.Errorf("Expected %+v giving zones, got %+v", tc.expectedGivingZones, actualGivingZones)
			}
			if !reflect.DeepEqual(actualReceivingZones, tc.expectedReceivingZones) {
				t.Errorf("Expected %+v receiving zones, got %+v", tc.expectedReceivingZones, actualReceivingZones)
			}
		})
	}
}

func Test_getHintsByZone(t *testing.T) {
	testCases := []struct {
		name                 string
		slice                discovery.EndpointSlice
		allocatedHintsByZone EndpointZoneInfo
		allocations          map[string]allocation
		expectedHintsByZone  map[string]int
	}{{
		name:                 "empty",
		slice:                discovery.EndpointSlice{},
		allocations:          map[string]allocation{},
		allocatedHintsByZone: EndpointZoneInfo{},
		expectedHintsByZone:  map[string]int{},
	}, {
		name: "single zone hint",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{{
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{"zone-a": 1},
		expectedHintsByZone: map[string]int{
			"zone-a": 1,
		},
	}, {
		name: "single zone hint with 1 unready endpoint and 1 unknown endpoint",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{{
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			}, {
				Zone:       pointer.String("zone-a"),
				Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
			}, {
				Zone:  pointer.String("zone-a"),
				Hints: &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
			}},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{"zone-a": 1},
		expectedHintsByZone: map[string]int{
			"zone-a": 1,
		},
	}, {
		name: "multiple zone hints",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
				{
					Zone:       pointer.String("zone-b"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-b"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
			},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
			"zone-b": {maximum: 3},
			"zone-c": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{"zone-a": 1, "zone-b": 1, "zone-c": 1},
		expectedHintsByZone: map[string]int{
			"zone-a": 1,
			"zone-b": 2,
		},
	}, {
		name: "invalid by zones that no longer requires any allocations",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-non-existent"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
			},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{"zone-a": 1, "zone-b": 1, "zone-c": 1},
		expectedHintsByZone:  nil,
	}, {
		name: "invalid by endpoints with nil hints",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{
				{
					Zone:       pointer.String("zone-a"),
					Hints:      nil,
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
			},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{},
		expectedHintsByZone:  nil,
	}, {
		name: "invalid by endpoint with no hints",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
			},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 3},
		},
		allocatedHintsByZone: EndpointZoneInfo{},
		expectedHintsByZone:  nil,
	}, {
		name: "invalid by hints that would make minimum allocations impossible",
		slice: discovery.EndpointSlice{
			Endpoints: []discovery.Endpoint{
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
				{
					Zone:       pointer.String("zone-a"),
					Hints:      &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: "zone-a"}}},
					Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				},
			},
		},
		allocations: map[string]allocation{
			"zone-a": {maximum: 2},
		},
		allocatedHintsByZone: EndpointZoneInfo{"zone-a": 1},
		expectedHintsByZone:  nil,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualHintsByZone := getHintsByZone(&tc.slice, tc.allocatedHintsByZone, tc.allocations)

			if !reflect.DeepEqual(actualHintsByZone, tc.expectedHintsByZone) {
				// %#v for distinguishing between nil and empty map
				t.Errorf("Expected %#v hints by zones, got %#v", tc.expectedHintsByZone, actualHintsByZone)
			}
		})
	}
}
