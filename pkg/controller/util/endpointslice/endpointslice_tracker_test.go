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

package endpointslice

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestEndpointSliceTrackerUpdate(t *testing.T) {
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "example-1",
			Namespace:  "ns1",
			UID:        "original",
			Generation: 1,
			Labels:     map[string]string{discovery.LabelServiceName: "svc1"},
		},
	}

	epSlice1DifferentNS := epSlice1.DeepCopy()
	epSlice1DifferentNS.Namespace = "ns2"
	epSlice1DifferentNS.UID = "diff-ns"

	epSlice1DifferentService := epSlice1.DeepCopy()
	epSlice1DifferentService.Labels[discovery.LabelServiceName] = "svc2"
	epSlice1DifferentService.UID = "diff-svc"

	epSlice1NewerGen := epSlice1.DeepCopy()
	epSlice1NewerGen.Generation = 2

	testCases := map[string]struct {
		updateParam      *discovery.EndpointSlice
		checksParam      *discovery.EndpointSlice
		expectHas        bool
		expectShouldSync bool
		expectGeneration int64
	}{
		"same slice": {
			updateParam:      epSlice1,
			checksParam:      epSlice1,
			expectHas:        true,
			expectShouldSync: false,
			expectGeneration: epSlice1.Generation,
		},
		"different namespace": {
			updateParam:      epSlice1,
			checksParam:      epSlice1DifferentNS,
			expectHas:        false,
			expectShouldSync: true,
			expectGeneration: epSlice1.Generation,
		},
		"different service": {
			updateParam:      epSlice1,
			checksParam:      epSlice1DifferentService,
			expectHas:        false,
			expectShouldSync: true,
			expectGeneration: epSlice1.Generation,
		},
		"newer generation": {
			updateParam:      epSlice1,
			checksParam:      epSlice1NewerGen,
			expectHas:        true,
			expectShouldSync: true,
			expectGeneration: epSlice1.Generation,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := NewEndpointSliceTracker()
			esTracker.Update(tc.updateParam)
			if esTracker.Has(tc.checksParam) != tc.expectHas {
				t.Errorf("tc.tracker.Has(%+v) == %t, expected %t", tc.checksParam, esTracker.Has(tc.checksParam), tc.expectHas)
			}
			if esTracker.ShouldSync(tc.checksParam) != tc.expectShouldSync {
				t.Errorf("tc.tracker.ShouldSync(%+v) == %t, expected %t", tc.checksParam, esTracker.ShouldSync(tc.checksParam), tc.expectShouldSync)
			}
			serviceNN := types.NamespacedName{Namespace: epSlice1.Namespace, Name: "svc1"}
			gfs, ok := esTracker.generationsByService[serviceNN]
			if !ok {
				t.Fatalf("expected tracker to have generations for %s Service", serviceNN.Name)
			}
			generation, ok := gfs[epSlice1.UID]
			if !ok {
				t.Fatalf("expected tracker to have generation for %s EndpointSlice", epSlice1.Name)
			}
			if tc.expectGeneration != generation {
				t.Fatalf("expected generation to be %d, got %d", tc.expectGeneration, generation)
			}
		})
	}
}

func TestEndpointSliceTrackerStaleSlices(t *testing.T) {
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "example-1",
			Namespace:  "ns1",
			UID:        "original",
			Generation: 1,
			Labels:     map[string]string{discovery.LabelServiceName: "svc1"},
		},
	}

	epSlice1NewerGen := epSlice1.DeepCopy()
	epSlice1NewerGen.Generation = 2

	epTerminatingSlice := epSlice1.DeepCopy()
	now := metav1.Now()
	epTerminatingSlice.DeletionTimestamp = &now

	testCases := []struct {
		name         string
		tracker      *EndpointSliceTracker
		serviceParam *v1.Service
		slicesParam  []*discovery.EndpointSlice
		expectNewer  bool
	}{{
		name: "empty tracker",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{},
		expectNewer:  false,
	}, {
		name: "empty slices",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{},
		expectNewer:  false,
	}, {
		name: "matching slices",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {
					epSlice1.UID: epSlice1.Generation,
				},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{epSlice1},
		expectNewer:  false,
	}, {
		name: "newer slice in tracker",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {
					epSlice1.UID: epSlice1NewerGen.Generation,
				},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{epSlice1},
		expectNewer:  true,
	}, {
		name: "newer slice in params",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {
					epSlice1.UID: epSlice1.Generation,
				},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{epSlice1NewerGen},
		expectNewer:  false,
	}, {
		name: "slice in params is expected to be deleted",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {
					epSlice1.UID: deletionExpected,
				},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{epSlice1},
		expectNewer:  true,
	}, {
		name: "slice in tracker but not in params",
		tracker: &EndpointSliceTracker{
			generationsByService: map[types.NamespacedName]GenerationsBySlice{
				{Name: "svc1", Namespace: "ns1"}: {
					epSlice1.UID: epSlice1.Generation,
				},
			},
		},
		serviceParam: &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "svc1", Namespace: "ns1"}},
		slicesParam:  []*discovery.EndpointSlice{},
		expectNewer:  true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualNewer := tc.tracker.StaleSlices(tc.serviceParam, tc.slicesParam)
			if actualNewer != tc.expectNewer {
				t.Errorf("Expected %t, got %t", tc.expectNewer, actualNewer)
			}
		})
	}
}
func TestEndpointSliceTrackerDeletion(t *testing.T) {
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "example-1",
			Namespace:  "ns1",
			UID:        "original",
			Generation: 1,
			Labels:     map[string]string{discovery.LabelServiceName: "svc1"},
		},
	}

	epSlice1DifferentNS := epSlice1.DeepCopy()
	epSlice1DifferentNS.Namespace = "ns2"
	epSlice1DifferentNS.UID = "diff-ns"

	epSlice1DifferentService := epSlice1.DeepCopy()
	epSlice1DifferentService.Labels[discovery.LabelServiceName] = "svc2"
	epSlice1DifferentService.UID = "diff-svc"

	epSlice1NewerGen := epSlice1.DeepCopy()
	epSlice1NewerGen.Generation = 2

	testCases := map[string]struct {
		expectDeletionParam        *discovery.EndpointSlice
		checksParam                *discovery.EndpointSlice
		deleteParam                *discovery.EndpointSlice
		expectHas                  bool
		expectShouldSync           bool
		expectedHandleDeletionResp bool
	}{
		"same slice": {
			expectDeletionParam:        epSlice1,
			checksParam:                epSlice1,
			deleteParam:                epSlice1,
			expectHas:                  true,
			expectShouldSync:           true,
			expectedHandleDeletionResp: true,
		},
		"different namespace": {
			expectDeletionParam:        epSlice1DifferentNS,
			checksParam:                epSlice1DifferentNS,
			deleteParam:                epSlice1DifferentNS,
			expectHas:                  true,
			expectShouldSync:           true,
			expectedHandleDeletionResp: false,
		},
		"different namespace, check original ep slice": {
			expectDeletionParam:        epSlice1DifferentNS,
			checksParam:                epSlice1,
			deleteParam:                epSlice1DifferentNS,
			expectHas:                  true,
			expectShouldSync:           false,
			expectedHandleDeletionResp: false,
		},
		"different service": {
			expectDeletionParam:        epSlice1DifferentService,
			checksParam:                epSlice1DifferentService,
			deleteParam:                epSlice1DifferentService,
			expectHas:                  true,
			expectShouldSync:           true,
			expectedHandleDeletionResp: false,
		},
		"expectDelete different service, check original ep slice, delete original": {
			expectDeletionParam:        epSlice1DifferentService,
			checksParam:                epSlice1,
			deleteParam:                epSlice1,
			expectHas:                  true,
			expectShouldSync:           false,
			expectedHandleDeletionResp: false,
		},
		"different generation": {
			expectDeletionParam:        epSlice1NewerGen,
			checksParam:                epSlice1NewerGen,
			deleteParam:                epSlice1NewerGen,
			expectHas:                  true,
			expectShouldSync:           true,
			expectedHandleDeletionResp: true,
		},
		"expectDelete different generation, check original ep slice, delete original": {
			expectDeletionParam:        epSlice1NewerGen,
			checksParam:                epSlice1,
			deleteParam:                epSlice1,
			expectHas:                  true,
			expectShouldSync:           true,
			expectedHandleDeletionResp: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := NewEndpointSliceTracker()
			esTracker.Update(epSlice1)

			esTracker.ExpectDeletion(tc.expectDeletionParam)
			if esTracker.Has(tc.checksParam) != tc.expectHas {
				t.Errorf("esTracker.Has(%+v) == %t, expected %t", tc.checksParam, esTracker.Has(tc.checksParam), tc.expectHas)
			}
			if esTracker.ShouldSync(tc.checksParam) != tc.expectShouldSync {
				t.Errorf("esTracker.ShouldSync(%+v) == %t, expected %t", tc.checksParam, esTracker.ShouldSync(tc.checksParam), tc.expectShouldSync)
			}
			if esTracker.HandleDeletion(epSlice1) != tc.expectedHandleDeletionResp {
				t.Errorf("esTracker.ShouldSync(%+v) == %t, expected %t", epSlice1, esTracker.HandleDeletion(epSlice1), tc.expectedHandleDeletionResp)
			}
			if esTracker.Has(epSlice1) != false {
				t.Errorf("esTracker.Has(%+v) == %t, expected false", epSlice1, esTracker.Has(epSlice1))
			}

		})
	}
}

func TestEndpointSliceTrackerDeleteService(t *testing.T) {
	svcName1, svcNS1 := "svc1", "ns1"
	svcName2, svcNS2 := "svc2", "ns2"
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "example-1",
			Namespace:  svcNS1,
			Generation: 1,
			Labels:     map[string]string{discovery.LabelServiceName: svcName1},
		},
	}

	testCases := map[string]struct {
		updateParam        *discovery.EndpointSlice
		deleteServiceParam *types.NamespacedName
		expectHas          bool
		expectShouldSync   bool
		expectGeneration   int64
	}{
		"same service": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS1, Name: svcName1},
			expectHas:          false,
			expectShouldSync:   true,
		},
		"different namespace": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS2, Name: svcName1},
			expectHas:          true,
			expectShouldSync:   false,
			expectGeneration:   epSlice1.Generation,
		},
		"different service": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS1, Name: svcName2},
			expectHas:          true,
			expectShouldSync:   false,
			expectGeneration:   epSlice1.Generation,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := NewEndpointSliceTracker()
			esTracker.Update(tc.updateParam)
			esTracker.DeleteService(tc.deleteServiceParam.Namespace, tc.deleteServiceParam.Name)
			if esTracker.Has(tc.updateParam) != tc.expectHas {
				t.Errorf("tc.tracker.Has(%+v) == %t, expected %t", tc.updateParam, esTracker.Has(tc.updateParam), tc.expectHas)
			}
			if esTracker.ShouldSync(tc.updateParam) != tc.expectShouldSync {
				t.Errorf("tc.tracker.ShouldSync(%+v) == %t, expected %t", tc.updateParam, esTracker.ShouldSync(tc.updateParam), tc.expectShouldSync)
			}
			if tc.expectGeneration != 0 {
				serviceNN := types.NamespacedName{Namespace: epSlice1.Namespace, Name: "svc1"}
				gfs, ok := esTracker.generationsByService[serviceNN]
				if !ok {
					t.Fatalf("expected tracker to have status for %s Service", serviceNN.Name)
				}
				generation, ok := gfs[epSlice1.UID]
				if !ok {
					t.Fatalf("expected tracker to have generation for %s EndpointSlice", epSlice1.Name)
				}
				if tc.expectGeneration != generation {
					t.Fatalf("expected generation to be %d, got %d", tc.expectGeneration, generation)
				}
			}
		})
	}
}
