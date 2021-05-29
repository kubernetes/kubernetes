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
	"fmt"
	"testing"

	discovery "k8s.io/api/discovery/v1"
)

func TestGetTotalEndpoints(t *testing.T) {
	testCases := []struct {
		name          string
		si            *SliceInfo
		expectedTotal int
	}{{
		name:          "empty",
		si:            &SliceInfo{},
		expectedTotal: 0,
	}, {
		name: "empty slice",
		si: &SliceInfo{
			ToCreate: []*discovery.EndpointSlice{sliceWithNEndpoints(0)},
		},
		expectedTotal: 0,
	}, {
		name: "multiple slices",
		si: &SliceInfo{
			ToCreate: []*discovery.EndpointSlice{sliceWithNEndpoints(15), sliceWithNEndpoints(8)},
		},
		expectedTotal: 23,
	}, {
		name: "slices for all",
		si: &SliceInfo{
			ToCreate:  []*discovery.EndpointSlice{sliceWithNEndpoints(15), sliceWithNEndpoints(8)},
			ToUpdate:  []*discovery.EndpointSlice{sliceWithNEndpoints(2)},
			Unchanged: []*discovery.EndpointSlice{sliceWithNEndpoints(100), sliceWithNEndpoints(90)},
		},
		expectedTotal: 215,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualTotal := tc.si.getTotalEndpoints()
			if actualTotal != tc.expectedTotal {
				t.Errorf("Expected %d, got %d", tc.expectedTotal, actualTotal)
			}
		})
	}
}

// helpers

func sliceWithNEndpoints(n int) *discovery.EndpointSlice {
	endpoints := []discovery.Endpoint{}

	for i := 0; i < n; i++ {
		endpoints = append(endpoints, discovery.Endpoint{Addresses: []string{fmt.Sprintf("10.1.2.%d", i)}})
	}

	return &discovery.EndpointSlice{
		Endpoints: endpoints,
	}
}
