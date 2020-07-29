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

	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestEndpointSliceTrackerUpdate(t *testing.T) {
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "example-1",
			Namespace:       "ns1",
			ResourceVersion: "rv1",
			Labels:          map[string]string{discovery.LabelServiceName: "svc1"},
		},
	}

	epSlice1DifferentNS := epSlice1.DeepCopy()
	epSlice1DifferentNS.Namespace = "ns2"

	epSlice1DifferentService := epSlice1.DeepCopy()
	epSlice1DifferentService.Labels[discovery.LabelServiceName] = "svc2"

	epSlice1DifferentRV := epSlice1.DeepCopy()
	epSlice1DifferentRV.ResourceVersion = "rv2"

	testCases := map[string]struct {
		updateParam           *discovery.EndpointSlice
		checksParam           *discovery.EndpointSlice
		expectHas             bool
		expectStale           bool
		expectResourceVersion string
	}{
		"same slice": {
			updateParam:           epSlice1,
			checksParam:           epSlice1,
			expectHas:             true,
			expectStale:           false,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
		"different namespace": {
			updateParam:           epSlice1,
			checksParam:           epSlice1DifferentNS,
			expectHas:             false,
			expectStale:           true,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
		"different service": {
			updateParam:           epSlice1,
			checksParam:           epSlice1DifferentService,
			expectHas:             false,
			expectStale:           true,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
		"different resource version": {
			updateParam:           epSlice1,
			checksParam:           epSlice1DifferentRV,
			expectHas:             true,
			expectStale:           true,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := newEndpointSliceTracker()
			esTracker.Update(tc.updateParam)
			if esTracker.Has(tc.checksParam) != tc.expectHas {
				t.Errorf("tc.tracker.Has(%+v) == %t, expected %t", tc.checksParam, esTracker.Has(tc.checksParam), tc.expectHas)
			}
			if esTracker.Stale(tc.checksParam) != tc.expectStale {
				t.Errorf("tc.tracker.Stale(%+v) == %t, expected %t", tc.checksParam, esTracker.Stale(tc.checksParam), tc.expectStale)
			}
			serviceNN := types.NamespacedName{Namespace: epSlice1.Namespace, Name: "svc1"}
			ss, ok := esTracker.statusByService[serviceNN]
			if !ok {
				t.Fatalf("expected tracker to have status for %s Service", serviceNN.Name)
			}
			sliceStatus, ok := ss.statusBySlice[epSlice1.Name]
			if !ok {
				t.Fatalf("expected tracker to have status for %s EndpointSlice", epSlice1.Name)
			}
			if tc.expectResourceVersion != sliceStatus.resourceVersion {
				t.Fatalf("expected resource version to be %s, got %s", tc.expectResourceVersion, sliceStatus.resourceVersion)
			}
		})
	}
}

func TestEndpointSliceTrackerDelete(t *testing.T) {
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "example-1",
			Namespace:       "ns1",
			ResourceVersion: "rv1",
			Labels:          map[string]string{discovery.LabelServiceName: "svc1"},
		},
	}

	epSlice1DifferentNS := epSlice1.DeepCopy()
	epSlice1DifferentNS.Namespace = "ns2"

	epSlice1DifferentService := epSlice1.DeepCopy()
	epSlice1DifferentService.Labels[discovery.LabelServiceName] = "svc2"

	epSlice1DifferentRV := epSlice1.DeepCopy()
	epSlice1DifferentRV.ResourceVersion = "rv2"

	testCases := map[string]struct {
		deleteParam *discovery.EndpointSlice
		checksParam *discovery.EndpointSlice
		expectHas   bool
		expectStale bool
	}{
		"same slice": {
			deleteParam: epSlice1,
			checksParam: epSlice1,
			expectHas:   false,
			expectStale: true,
		},
		"different namespace": {
			deleteParam: epSlice1DifferentNS,
			checksParam: epSlice1DifferentNS,
			expectHas:   false,
			expectStale: true,
		},
		"different namespace, check original ep slice": {
			deleteParam: epSlice1DifferentNS,
			checksParam: epSlice1,
			expectHas:   true,
			expectStale: false,
		},
		"different service": {
			deleteParam: epSlice1DifferentService,
			checksParam: epSlice1DifferentService,
			expectHas:   false,
			expectStale: true,
		},
		"different service, check original ep slice": {
			deleteParam: epSlice1DifferentService,
			checksParam: epSlice1,
			expectHas:   true,
			expectStale: false,
		},
		"different resource version": {
			deleteParam: epSlice1DifferentRV,
			checksParam: epSlice1DifferentRV,
			expectHas:   false,
			expectStale: true,
		},
		"different resource version, check original ep slice": {
			deleteParam: epSlice1DifferentRV,
			checksParam: epSlice1,
			expectHas:   false,
			expectStale: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := newEndpointSliceTracker()
			esTracker.Update(epSlice1)

			esTracker.Delete(tc.deleteParam)
			if esTracker.Has(tc.checksParam) != tc.expectHas {
				t.Errorf("esTracker.Has(%+v) == %t, expected %t", tc.checksParam, esTracker.Has(tc.checksParam), tc.expectHas)
			}
			if esTracker.Stale(tc.checksParam) != tc.expectStale {
				t.Errorf("esTracker.Stale(%+v) == %t, expected %t", tc.checksParam, esTracker.Stale(tc.checksParam), tc.expectStale)
			}
		})
	}
}

func TestEndpointSliceTrackerDeleteService(t *testing.T) {
	svcName1, svcNS1 := "svc1", "ns1"
	svcName2, svcNS2 := "svc2", "ns2"
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "example-1",
			Namespace:       svcNS1,
			ResourceVersion: "rv1",
			Labels:          map[string]string{discovery.LabelServiceName: svcName1},
		},
	}

	testCases := map[string]struct {
		updateParam           *discovery.EndpointSlice
		deleteServiceParam    *types.NamespacedName
		expectHas             bool
		expectStale           bool
		expectResourceVersion string
	}{
		"same service": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS1, Name: svcName1},
			expectHas:          false,
			expectStale:        true,
		},
		"different namespace": {
			updateParam:           epSlice1,
			deleteServiceParam:    &types.NamespacedName{Namespace: svcNS2, Name: svcName1},
			expectHas:             true,
			expectStale:           false,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
		"different service": {
			updateParam:           epSlice1,
			deleteServiceParam:    &types.NamespacedName{Namespace: svcNS1, Name: svcName2},
			expectHas:             true,
			expectStale:           false,
			expectResourceVersion: epSlice1.ResourceVersion,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			esTracker := newEndpointSliceTracker()
			esTracker.Update(tc.updateParam)
			esTracker.DeleteService(tc.deleteServiceParam.Namespace, tc.deleteServiceParam.Name)
			if esTracker.Has(tc.updateParam) != tc.expectHas {
				t.Errorf("tc.tracker.Has(%+v) == %t, expected %t", tc.updateParam, esTracker.Has(tc.updateParam), tc.expectHas)
			}
			if esTracker.Stale(tc.updateParam) != tc.expectStale {
				t.Errorf("tc.tracker.Stale(%+v) == %t, expected %t", tc.updateParam, esTracker.Stale(tc.updateParam), tc.expectStale)
			}
			if tc.expectResourceVersion != "" {
				serviceNN := types.NamespacedName{Namespace: epSlice1.Namespace, Name: "svc1"}
				ss, ok := esTracker.statusByService[serviceNN]
				if !ok {
					t.Fatalf("expected tracker to have status for %s Service", serviceNN.Name)
				}
				sliceStatus, ok := ss.statusBySlice[epSlice1.Name]
				if !ok {
					t.Fatalf("expected tracker to have status for %s EndpointSlice", epSlice1.Name)
				}
				if tc.expectResourceVersion != sliceStatus.resourceVersion {
					t.Fatalf("expected resource version to be %s, got %s", tc.expectResourceVersion, sliceStatus.resourceVersion)
				}
			}
		})
	}
}

func TestServiceCacheOutdated(t *testing.T) {
	testCases := []struct {
		testName        string
		svcNamespace    string
		svcName         string
		statusByService map[types.NamespacedName]*serviceStatus
		expectOutdated  bool
	}{{
		testName:       "empty statusByService",
		svcNamespace:   "foo",
		svcName:        "bar",
		expectOutdated: false,
	}, {
		testName:     "statusByService with no slices",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{},
			},
		},
		expectOutdated: false,
	}, {
		testName:     "statusByService with one slice out of date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: false},
				},
			},
		},
		expectOutdated: true,
	}, {
		testName:     "statusByService with one slice out of date, different namespace",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo2", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: false},
				},
			},
		},
		expectOutdated: false,
	}, {
		testName:     "statusByService with one slice up to date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: true},
				},
			},
		},
		expectOutdated: false,
	}, {
		testName:     "statusByService with one slice up to date and one out of date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: true},
					"slice-2": {cacheUpdated: false},
				},
			},
		},
		expectOutdated: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			est := newEndpointSliceTracker()
			est.statusByService = tc.statusByService
			actualOutdated := est.ServiceCacheOutdated(tc.svcNamespace, tc.svcName)

			if tc.expectOutdated != actualOutdated {
				t.Errorf("Expected ServiceCacheOutdated() to return %t, got %t", tc.expectOutdated, actualOutdated)
			}
		})
	}
}

func TestMarkCacheUpdated(t *testing.T) {
	testCases := []struct {
		testName             string
		svcNamespace         string
		svcName              string
		statusByService      map[types.NamespacedName]*serviceStatus
		expectOutdatedBefore bool
		endpointSlice        *discovery.EndpointSlice
		expectOutdatedAfter  bool
	}{{
		testName:             "empty statusByService",
		svcNamespace:         "foo",
		svcName:              "bar",
		expectOutdatedBefore: false,
		endpointSlice:        endpointSliceForService("slice-1", "foo", "bar"),
		expectOutdatedAfter:  false,
	}, {
		testName:     "statusByService with no slices",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{},
			},
		},
		expectOutdatedBefore: false,
		endpointSlice:        endpointSliceForService("slice-1", "foo", "bar"),
		expectOutdatedAfter:  false,
	}, {
		testName:     "statusByService with one slice out of date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: false},
				},
			},
		},
		expectOutdatedBefore: true,
		endpointSlice:        endpointSliceForService("slice-1", "foo", "bar"),
		expectOutdatedAfter:  false,
	}, {
		testName:     "statusByService with one slice out of date, different namespace",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo2", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: false},
				},
			},
		},
		expectOutdatedBefore: false,
		endpointSlice:        endpointSliceForService("slice-1", "foo", "bar"),
		expectOutdatedAfter:  false,
	}, {
		testName:     "statusByService with one slice up to date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: true},
				},
			},
		},
		expectOutdatedBefore: false,
		endpointSlice:        endpointSliceForService("slice-1", "foo", "bar"),
		expectOutdatedAfter:  false,
	}, {
		testName:     "statusByService with one slice up to date and one out of date",
		svcNamespace: "foo",
		svcName:      "bar",
		statusByService: map[types.NamespacedName]*serviceStatus{
			{Namespace: "foo", Name: "bar"}: {
				statusBySlice: map[string]*sliceStatus{
					"slice-1": {cacheUpdated: true},
					"slice-2": {cacheUpdated: false},
				},
			},
		},
		expectOutdatedBefore: true,
		endpointSlice:        endpointSliceForService("slice-2", "foo", "bar"),
		expectOutdatedAfter:  false,
	}}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			est := newEndpointSliceTracker()
			est.statusByService = tc.statusByService
			actualOutdatedBefore := est.ServiceCacheOutdated(tc.svcNamespace, tc.svcName)

			if tc.expectOutdatedBefore != actualOutdatedBefore {
				t.Errorf("Expected ServiceCacheOutdated() to return %t before update, got %t", tc.expectOutdatedBefore, actualOutdatedBefore)
			}

			est.MarkCacheUpdated(tc.endpointSlice)

			actualOutdatedAfter := est.ServiceCacheOutdated(tc.svcNamespace, tc.svcName)

			if tc.expectOutdatedAfter != actualOutdatedAfter {
				t.Errorf("Expected ServiceCacheOutdated() to return %t after update, got %t", tc.expectOutdatedAfter, actualOutdatedAfter)
			}

		})
	}
}

func endpointSliceForService(name, namespace, svcName string) *discovery.EndpointSlice {
	return &discovery.EndpointSlice{ObjectMeta: metav1.ObjectMeta{
		Namespace: namespace,
		Name:      name,
		Labels: map[string]string{
			discovery.LabelServiceName: svcName,
		},
	}}
}
