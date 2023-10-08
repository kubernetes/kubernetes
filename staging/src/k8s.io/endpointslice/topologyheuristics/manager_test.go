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
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

// TestManager_PopulateHints will transition a service through various topology
// heuristics and ensure that PopulateHints handles all such transitions
// properly.
//
// Note: This test function only covers the PopulateHints of the Manager.
// Individual Heuristics have their own test functions for PopulateHints.
func TestManager_PopulateHints(t *testing.T) {
	//////////////////////////////////////////////////////////////////////////////
	// Step 1: Setup test variables.
	//////////////////////////////////////////////////////////////////////////////

	logger, _ := ktesting.NewTestContext(t)
	svcKey := "ns/svc1"
	addrType := discoveryv1.AddressTypeIPv4
	// Initialize heuristics.
	fooHeuristic := &fakeHeuristic{name: "Foo", zone: "zone1"}
	proportionalZoneCPUHeuristic := &fakeHeuristic{name: "ProportionalZoneCPU", zone: "zone2"}
	preferZoneHeuristic := &fakeHeuristic{name: "PreferZone", zone: "zone3"}
	topologies := []Heuristic{
		fooHeuristic,
		proportionalZoneCPUHeuristic,
		preferZoneHeuristic,
	}
	// Initialize heuristic manager.
	heuristicManager, _ := NewManager(logger, topologies, proportionalZoneCPUHeuristic.Name(), []string{})

	//////////////////////////////////////////////////////////////////////////////
	// Step 2: Define local helper methods for assertion
	//////////////////////////////////////////////////////////////////////////////

	// mustHaveCachedTopologyHints ensures that the given heuristic has cached
	// topology hints.
	mustHaveCachedTopologyHints := func(t *testing.T, heuristic *fakeHeuristic) {
		t.Helper()
		if !heuristic.hasCachedHints(svcKey, addrType) {
			t.Fatalf("Got no cached topology hints in heuristic %q; want cached hints", heuristic.Name())
		}
	}
	// mustNotHaveCachedTopologyHints ensures that the given heuristic DOES NOT
	// have cached topology hints.
	mustNotHaveCachedTopologyHints := func(t *testing.T, heuristic *fakeHeuristic) {
		t.Helper()
		if heuristic.hasCachedHints(svcKey, addrType) {
			t.Fatalf("Got cached topology hints in heuristic %q; want no cached hints", heuristic.Name())
		}
	}
	// newSliceInfo returns a newly created SliceInfo. We can't reuse the same
	// SliceInfo because PopulateHints may modify it.
	newSliceInfo := func() *SliceInfo {
		return &SliceInfo{
			ServiceKey:  "ns/svc1",
			AddressType: discoveryv1.AddressTypeIPv4,
			ToCreate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses: []string{"10.0.0.1"},
							Hints:     &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "randomZone"}}},
						},
					},
				},
			},
			ToUpdate: []*discoveryv1.EndpointSlice{
				{
					Endpoints: []discoveryv1.Endpoint{
						{
							Addresses: []string{"10.0.0.2"},
							Hints:     &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: "randomZone"}}},
						},
					},
				},
			},
		}
	}
	// checkHintsForSlicesToCreate ensures that gotSlicesToCreate contains
	// topology hints set to wantZone or nil if wantZone is empty.
	checkHintsForSlicesToCreate := func(t *testing.T, wantZone string, gotSlicesToCreate []*discoveryv1.EndpointSlice) {
		t.Helper()
		want := []*discoveryv1.EndpointSlice{
			{
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses: []string{"10.0.0.1"},
					},
				},
			},
		}
		if wantZone != "" {
			want[0].Endpoints[0].Hints = &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: wantZone}}}
		}
		if diff := cmp.Diff(want, gotSlicesToCreate); diff != "" {
			t.Fatalf("Unexpected diff in slicesToCreate: (-want, +got)\n%v", diff)
		}
	}
	// checkHintsForSlicesToCreate ensures that gotSlicesToCreate contains
	// topology hints set to wantZone or nil if wantZone is empty.
	checkHintsForSlicesToUpdate := func(t *testing.T, wantZone string, gotSlicesToCreate []*discoveryv1.EndpointSlice) {
		t.Helper()
		want := []*discoveryv1.EndpointSlice{
			{
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses: []string{"10.0.0.2"},
					},
				},
			},
		}
		if wantZone != "" {
			want[0].Endpoints[0].Hints = &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: wantZone}}}
		}
		if diff := cmp.Diff(want, gotSlicesToCreate); diff != "" {
			t.Fatalf("Unexpected diff in slicesToUpdate: (-want, +got)\n%v", diff)
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// Step 3: Invoke function under test and start assertions.
	//////////////////////////////////////////////////////////////////////////////

	// Initially, no topologies should have cached topology hints.
	mustNotHaveCachedTopologyHints(t, fooHeuristic)
	mustNotHaveCachedTopologyHints(t, proportionalZoneCPUHeuristic)
	mustNotHaveCachedTopologyHints(t, preferZoneHeuristic)

	// Change heuristic to "Foo".
	// Transition from empty -> "Foo".
	gotSlicesToCreate, gotSlicesToUpdate, _ := heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(fooHeuristic.Name()),
		newSliceInfo(),
	)
	checkHintsForSlicesToCreate(t, "zone1", gotSlicesToCreate)
	checkHintsForSlicesToUpdate(t, "zone1", gotSlicesToUpdate)
	mustHaveCachedTopologyHints(t, fooHeuristic) // Active heuristic
	mustNotHaveCachedTopologyHints(t, proportionalZoneCPUHeuristic)
	mustNotHaveCachedTopologyHints(t, preferZoneHeuristic)

	// Change heuristic to "ProportionalZoneCPU".
	// Transition from "Foo" -> "ProportionalZoneCPU".
	gotSlicesToCreate, gotSlicesToUpdate, _ = heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(proportionalZoneCPUHeuristic.Name()),
		newSliceInfo(),
	)
	checkHintsForSlicesToCreate(t, "zone2", gotSlicesToCreate)
	checkHintsForSlicesToUpdate(t, "zone2", gotSlicesToUpdate)
	mustNotHaveCachedTopologyHints(t, fooHeuristic)
	mustHaveCachedTopologyHints(t, proportionalZoneCPUHeuristic) // Active heuristic
	mustNotHaveCachedTopologyHints(t, preferZoneHeuristic)

	// Change heuristic to "PreferZone"
	// Transition from "ProportionalZoneCPU" -> "PreferZone".
	gotSlicesToCreate, gotSlicesToUpdate, _ = heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(preferZoneHeuristic.Name()),
		newSliceInfo(),
	)
	checkHintsForSlicesToCreate(t, "zone3", gotSlicesToCreate)
	checkHintsForSlicesToUpdate(t, "zone3", gotSlicesToUpdate)
	mustNotHaveCachedTopologyHints(t, fooHeuristic)
	mustNotHaveCachedTopologyHints(t, proportionalZoneCPUHeuristic)
	mustHaveCachedTopologyHints(t, preferZoneHeuristic) // Active heuristic

	// Change heuristic to "".
	// Transition from "PreferZone" -> "".
	//
	// This should remove the topology hints from all slices and clear cached
	// hints from all topology heuristics.
	gotSlicesToCreate, gotSlicesToUpdate, _ = heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(""),
		newSliceInfo(),
	)
	checkHintsForSlicesToCreate(t, "", gotSlicesToCreate)
	checkHintsForSlicesToUpdate(t, "", gotSlicesToUpdate)
	mustNotHaveCachedTopologyHints(t, fooHeuristic)
	mustNotHaveCachedTopologyHints(t, proportionalZoneCPUHeuristic)
	mustNotHaveCachedTopologyHints(t, preferZoneHeuristic)
}

func TestManager_ClearCachedHints(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	svcKey := "ns/svc1"
	addrType := discoveryv1.AddressTypeIPv4
	// Initialize heuristics.
	fooHeuristic := &fakeHeuristic{name: "Foo"}
	proportionalZoneCPUHeuristic := &fakeHeuristic{name: "ProportionalZoneCPU"}
	preferZoneHeuristic := &fakeHeuristic{name: "PreferZone"}
	topologies := []Heuristic{
		fooHeuristic,
		proportionalZoneCPUHeuristic,
		preferZoneHeuristic,
	}
	// Initialize heuristics manager.
	heuristicManager, _ := NewManager(logger, topologies, proportionalZoneCPUHeuristic.Name(), []string{})

	// Setup: Populate topology hints in "Foo" heuristic.
	heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(fooHeuristic.Name()),
		&SliceInfo{ServiceKey: "ns/svc1", AddressType: discoveryv1.AddressTypeIPv4},
	)
	// Ensure that topology hints were cached.
	if !fooHeuristic.hasCachedHints(svcKey, addrType) {
		t.Fatalf("Got no cached topology hints in heuristic %q; want cached hints", fooHeuristic.Name())
	}

	// Invoke function under test. This should remove the cached topology hints
	// from "Foo" heuristic.
	heuristicManager.ClearCachedHints(logger, svcKey, addrType)
	if fooHeuristic.hasCachedHints(svcKey, addrType) {
		t.Errorf("ClearCachedHints(): Got cached topology hints in heuristic %q; want no cached hints after invoking ClearCachedHints()", fooHeuristic.Name())
	}
}

func TestManager_activeHeuristicForService(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	fooHeuristic := &fakeHeuristic{name: "Foo"}
	proportionalZoneCPUHeuristic := &fakeHeuristic{name: "ProportionalZoneCPU"}
	barHeuristic := &fakeHeuristic{name: "Bar"}
	topologies := []Heuristic{
		fooHeuristic,
		proportionalZoneCPUHeuristic,
		barHeuristic,
	}
	heuristicManager, _ := NewManager(logger, topologies, proportionalZoneCPUHeuristic.Name(), []string{})

	// At this point, active heuristic for the service should not exist
	_, activeHeuristicExists := heuristicManager.activeHeuristicForService("ns/svc1", discoveryv1.AddressTypeIPv4)
	if activeHeuristicExists {
		t.Fatalf("activeHeuristicForService(): There should be no active heuristic before invoking PopulateHints")
	}

	// Invoke PopulateHints. Now active heuristic should exist.
	heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(barHeuristic.Name()),
		&SliceInfo{ServiceKey: "ns/svc1", AddressType: discoveryv1.AddressTypeIPv4},
	)
	activeHeuristic, activeHeuristicExists := heuristicManager.activeHeuristicForService("ns/svc1", discoveryv1.AddressTypeIPv4)
	if !activeHeuristicExists {
		t.Fatalf("activeHeuristicForService(): Active heuristic should exist because PopulateHints has been invoked")
	}
	if activeHeuristic.Name() != barHeuristic.Name() {
		t.Fatalf("activeHeuristicForService(): got heuristic name = %v, want heuristic name = %v", activeHeuristic.Name(), barHeuristic.Name())
	}

	// Invoke PopulateHints with a different heuristic. Previous heuristic should
	// get overwritten.
	heuristicManager.PopulateHints(
		logger,
		serviceWithTopologyAnnotation(fooHeuristic.Name()),
		&SliceInfo{ServiceKey: "ns/svc1", AddressType: discoveryv1.AddressTypeIPv4},
	)
	activeHeuristic, activeHeuristicExists = heuristicManager.activeHeuristicForService("ns/svc1", discoveryv1.AddressTypeIPv4)
	if !activeHeuristicExists {
		t.Fatalf("activeHeuristicForService(): Active heuristic should exist because PopulateHints has been invoked")
	}
	if activeHeuristic.Name() != fooHeuristic.Name() {
		t.Fatalf("activeHeuristicForService(): got heuristic name = %v, want heuristic name = %v", activeHeuristic.Name(), fooHeuristic.Name())
	}

	// Invoke ClearCachedHints. There should no longer be any active heuristic.
	heuristicManager.ClearCachedHints(logger, "ns/svc1", discoveryv1.AddressTypeIPv4)
	_, activeHeuristicExists = heuristicManager.activeHeuristicForService("ns/svc1", discoveryv1.AddressTypeIPv4)
	if activeHeuristicExists {
		t.Fatalf("activeHeuristicForService(): Active heuristic should not exist because ClearCachedHints has been invoked")
	}
}

func TestManager_desiredHeuristicForService(t *testing.T) {
	testCases := []struct {
		name string

		inputHeuristics []Heuristic
		service         *corev1.Service

		wantHeuristicExists  bool // Heuristic exists in the heuristic manager.
		wantHeuristicEnabled bool // Topology annotation indicates an enabled topology.
		wantHeuristicName    string
	}{
		{
			name: "annotation=Auto",
			inputHeuristics: []Heuristic{
				&fakeHeuristic{name: "Foo"},
				&fakeHeuristic{name: "ProportionalZoneCPU"},
				&fakeHeuristic{name: "Bar"},
			},
			service:              serviceWithTopologyAnnotation("Auto"),
			wantHeuristicExists:  true,
			wantHeuristicEnabled: true,
			wantHeuristicName:    "ProportionalZoneCPU",
		},
		{
			name: "annotation=Bar",
			inputHeuristics: []Heuristic{
				&fakeHeuristic{name: "Foo"},
				&fakeHeuristic{name: "ProportionalZoneCPU"},
				&fakeHeuristic{name: "Bar"},
			},
			service:              serviceWithTopologyAnnotation("Bar"),
			wantHeuristicExists:  true,
			wantHeuristicEnabled: true,
			wantHeuristicName:    "Bar",
		},
		{
			name: "annotation=disabled",
			inputHeuristics: []Heuristic{
				&fakeHeuristic{name: "Foo"},
				&fakeHeuristic{name: "ProportionalZoneCPU"},
				&fakeHeuristic{name: "Bar"},
				// Although a topology with this name 'disabled' exists, this is treated
				// specially and hence the result should be 'not found'.
				&fakeHeuristic{name: "disabled"},
			},
			service:              serviceWithTopologyAnnotation("disabled"),
			wantHeuristicExists:  false,
			wantHeuristicEnabled: false,
		},
		{
			name: "annotation=Random; heuristic with this name not initialized in manager",
			inputHeuristics: []Heuristic{
				&fakeHeuristic{name: "Foo"},
				&fakeHeuristic{name: "ProportionalZoneCPU"},
				&fakeHeuristic{name: "Bar"},
			},
			service:              serviceWithTopologyAnnotation("Random"),
			wantHeuristicExists:  false,
			wantHeuristicEnabled: true,
		},
		{
			name: "annotation=foo; case is matched when comparing topology name",
			inputHeuristics: []Heuristic{
				&fakeHeuristic{name: "Foo"}, // Capitalized.
				&fakeHeuristic{name: "ProportionalZoneCPU"},
				&fakeHeuristic{name: "Bar"},
			},
			service:              serviceWithTopologyAnnotation("foo"), // Not capitalized.
			wantHeuristicExists:  false,
			wantHeuristicEnabled: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			heuristicManager, _ := NewManager(logger, tc.inputHeuristics, "ProportionalZoneCPU", []string{})

			gotHeuristic, gotHeuristicEnabled, gotHeuristicExists := heuristicManager.desiredHeuristicForService(tc.service)
			if gotHeuristicExists != tc.wantHeuristicExists {
				t.Fatalf("desiredHeuristicForService(...): gotHeuristicExists=%v, wantHeuristicExists=%v", gotHeuristicExists, tc.wantHeuristicExists)
			}
			if gotHeuristicEnabled != tc.wantHeuristicEnabled {
				t.Fatalf("desiredHeuristicForService(...): gotHeuristicEnabled=%v, wantHeuristicEnabled=%v", gotHeuristicEnabled, tc.wantHeuristicEnabled)
			}
			if gotHeuristicExists && gotHeuristic.Name() != tc.wantHeuristicName {
				t.Errorf("desiredHeuristicForService(...): gotHeuristicName=%v, wantHeuristicName=%v", gotHeuristic.Name(), tc.wantHeuristicName)
			}
		})
	}
}

func TestHeuristicNameFromAnnotations(t *testing.T) {
	testCases := []struct {
		name        string
		annotations map[string]string

		wantEnabled       bool
		wantHeuristicName string
	}{
		{
			name:        "empty annotations",
			wantEnabled: false,
		},
		{
			name:        "different annotations",
			annotations: map[string]string{"topology-hints": "enabled"},
			wantEnabled: false,
		},
		{
			name:        "hints annotation = '' (empty)",
			annotations: map[string]string{corev1.DeprecatedAnnotationTopologyAwareHints: ""},
			wantEnabled: false,
		},
		{
			name:        "hints annotation = disabled",
			annotations: map[string]string{corev1.DeprecatedAnnotationTopologyAwareHints: "disabled"},
			wantEnabled: false,
		},
		{
			name:        "hints annotation = Disabled",
			annotations: map[string]string{corev1.DeprecatedAnnotationTopologyAwareHints: "Disabled"},
			wantEnabled: false,
		},
		{
			name:              "hints annotation = auto",
			annotations:       map[string]string{corev1.DeprecatedAnnotationTopologyAwareHints: "auto"},
			wantEnabled:       true,
			wantHeuristicName: "auto",
		},
		{
			name:        "hints annotation = FooTopology; hints annotation does not support annotations besides Auto",
			annotations: map[string]string{corev1.DeprecatedAnnotationTopologyAwareHints: "FooTopology"},
			wantEnabled: false,
		},
		{
			name:        "mode annotation = '' (empty)",
			annotations: map[string]string{corev1.AnnotationTopologyMode: ""},
			wantEnabled: false,
		},
		{
			name:        "mode annotation = disabled",
			annotations: map[string]string{corev1.AnnotationTopologyMode: "disabled"},
			wantEnabled: false,
		},
		{
			name:        "mode annotation = Disabled",
			annotations: map[string]string{corev1.AnnotationTopologyMode: "Disabled"},
			wantEnabled: false,
		},
		{
			name:              "mode annotation = auto",
			annotations:       map[string]string{corev1.AnnotationTopologyMode: "auto"},
			wantEnabled:       true,
			wantHeuristicName: "auto",
		},
		{
			name:              "mode annotation = FooTopology",
			annotations:       map[string]string{corev1.AnnotationTopologyMode: "FooTopology"},
			wantEnabled:       true,
			wantHeuristicName: "FooTopology",
		},
		{
			name:              "mode annotation = disabled, hints annotation = auto",
			annotations:       map[string]string{corev1.AnnotationTopologyMode: "disabled", corev1.DeprecatedAnnotationTopologyAwareHints: "auto"},
			wantEnabled:       true,
			wantHeuristicName: "auto",
		},
		{
			name:        "mode annotation = auto, hints annotation = disabled",
			annotations: map[string]string{corev1.AnnotationTopologyMode: "auto", corev1.DeprecatedAnnotationTopologyAwareHints: "disabled"},
			wantEnabled: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotTopologyName, gotEnabled := HeuristicNameFromAnnotations(tc.annotations)
			if gotEnabled != tc.wantEnabled {
				t.Fatalf("HeuristicNameFromAnnotations(...) returned unexpected difference on whether topology is enabled; gotHeuristicEnabled=%v, wantHeuristicEnabled=%v", gotEnabled, tc.wantEnabled)
			}
			if tc.wantEnabled && gotTopologyName != tc.wantHeuristicName {
				t.Errorf("HeuristicNameFromAnnotations(...) returned unexpected topology name; got=%v, want=%v", gotTopologyName, tc.wantHeuristicName)
			}
		})
	}
}

// serviceWithTopologyAnnotation returns a Serivce with the topology annotation
// set to topologyAnnotationValue.
func serviceWithTopologyAnnotation(topologyAnnotationValue string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
			corev1.AnnotationTopologyMode: topologyAnnotationValue,
		}},
	}
}

type fakeHeuristic struct {
	// name is the value returned by the Name() function for the heuristic.
	name string
	// zone is the zone value which will be populated as the topology hint in all
	// returned value.
	zone string

	mu    sync.Mutex
	hints map[string]map[discoveryv1.AddressType]bool
}

func (t *fakeHeuristic) Name() string {
	return t.name
}

func (t *fakeHeuristic) PopulateHints(logger klog.Logger, si *SliceInfo) (slicesToCreate []*discoveryv1.EndpointSlice, slicesToUpdate []*discoveryv1.EndpointSlice, events []*EventBuilder) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.hints == nil {
		t.hints = make(map[string]map[discoveryv1.AddressType]bool)
	}
	if _, ok := t.hints[si.ServiceKey]; !ok {
		t.hints[si.ServiceKey] = make(map[discoveryv1.AddressType]bool)
	}

	// Add hints to local cache.
	t.hints[si.ServiceKey][si.AddressType] = true

	// Populate slices with fixed zone hints.
	for _, slice := range si.ToCreate {
		for i := range slice.Endpoints {
			slice.Endpoints[i].Hints = &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: t.zone}}}
		}
	}
	for _, slice := range si.ToUpdate {
		for i := range slice.Endpoints {
			slice.Endpoints[i].Hints = &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: t.zone}}}
		}
	}

	return si.ToCreate, si.ToUpdate, nil
}

func (t *fakeHeuristic) ClearCachedHints(logger klog.Logger, serviceKey string, addrType discoveryv1.AddressType) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.hints == nil {
		t.hints = make(map[string]map[discoveryv1.AddressType]bool)
	}
	if _, ok := t.hints[serviceKey]; !ok {
		t.hints[serviceKey] = make(map[discoveryv1.AddressType]bool)
	}

	t.hints[serviceKey][addrType] = false
}

// hasCachedHints returns true if the heuristic has any cached hints for the
// [serviceKey and addrType].
func (t *fakeHeuristic) hasCachedHints(serviceKey string, addrType discoveryv1.AddressType) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.hints == nil {
		t.hints = make(map[string]map[discoveryv1.AddressType]bool)
	}
	if _, ok := t.hints[serviceKey]; !ok {
		t.hints[serviceKey] = make(map[discoveryv1.AddressType]bool)
	}

	val, ok := t.hints[serviceKey][addrType]
	return val && ok
}
