/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	"testing"

	"github.com/stretchr/testify/assert"

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
		updateParam                     *discovery.EndpointSlice
		checksParam                     *discovery.EndpointSlice
		expectHas                       bool
		expectStale                     bool
		expectResourceVersionsByService map[types.NamespacedName]endpointSliceResourceVersions
	}{
		"same slice": {
			updateParam: epSlice1,
			checksParam: epSlice1,
			expectHas:   true,
			expectStale: false,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
		},
		"different namespace": {
			updateParam: epSlice1,
			checksParam: epSlice1DifferentNS,
			expectHas:   false,
			expectStale: true,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
		},
		"different service": {
			updateParam: epSlice1,
			checksParam: epSlice1DifferentService,
			expectHas:   false,
			expectStale: true,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
		},
		"different resource version": {
			updateParam: epSlice1,
			checksParam: epSlice1DifferentRV,
			expectHas:   true,
			expectStale: true,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
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
			assert.Equal(t, tc.expectResourceVersionsByService, esTracker.resourceVersionsByService)
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
		updateParam                     *discovery.EndpointSlice
		deleteServiceParam              *types.NamespacedName
		expectHas                       bool
		expectStale                     bool
		expectResourceVersionsByService map[types.NamespacedName]endpointSliceResourceVersions
	}{
		"same service": {
			updateParam:                     epSlice1,
			deleteServiceParam:              &types.NamespacedName{Namespace: svcNS1, Name: svcName1},
			expectHas:                       false,
			expectStale:                     true,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{},
		},
		"different namespace": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS2, Name: svcName1},
			expectHas:          true,
			expectStale:        false,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
		},
		"different service": {
			updateParam:        epSlice1,
			deleteServiceParam: &types.NamespacedName{Namespace: svcNS1, Name: svcName2},
			expectHas:          true,
			expectStale:        false,
			expectResourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{
				{Namespace: epSlice1.Namespace, Name: "svc1"}: {
					epSlice1.Name: epSlice1.ResourceVersion,
				},
			},
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
			assert.Equal(t, tc.expectResourceVersionsByService, esTracker.resourceVersionsByService)
		})
	}
}
