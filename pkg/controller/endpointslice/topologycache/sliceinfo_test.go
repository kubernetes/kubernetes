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
	"k8s.io/utils/pointer"
)

func Test_getTotalReadyEndpoints(t *testing.T) {
	testCases := []struct {
		name               string
		si                 *SliceInfo
		expectedTotalReady int
		expectedTotal      int // this is really just testing the test helper
	}{{
		name:          "empty",
		si:            &SliceInfo{},
		expectedTotal: 0,
	}, {
		name: "empty slice",
		si: &SliceInfo{
			ToCreate: []*discovery.EndpointSlice{sliceWithNEndpoints(0, 0)},
		},
		expectedTotalReady: 0,
		expectedTotal:      0,
	}, {
		name: "multiple slices",
		si: &SliceInfo{
			ToCreate: []*discovery.EndpointSlice{sliceWithNEndpoints(15, 0), sliceWithNEndpoints(8, 0)},
		},
		expectedTotalReady: 23,
		expectedTotal:      23,
	}, {
		name: "slices for all",
		si: &SliceInfo{
			ToCreate:  []*discovery.EndpointSlice{sliceWithNEndpoints(15, 0), sliceWithNEndpoints(8, 0)},
			ToUpdate:  []*discovery.EndpointSlice{sliceWithNEndpoints(2, 0)},
			Unchanged: []*discovery.EndpointSlice{sliceWithNEndpoints(100, 0), sliceWithNEndpoints(90, 0)},
		},
		expectedTotalReady: 215,
		expectedTotal:      215,
	}, {
		name: "slices for all with some unready",
		si: &SliceInfo{
			ToCreate:  []*discovery.EndpointSlice{sliceWithNEndpoints(15, 3), sliceWithNEndpoints(5, 4)},
			ToUpdate:  []*discovery.EndpointSlice{sliceWithNEndpoints(3, 8)},
			Unchanged: []*discovery.EndpointSlice{sliceWithNEndpoints(98, 2), sliceWithNEndpoints(90, 6)},
		},
		expectedTotalReady: 211,
		expectedTotal:      234,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualTotal := countEndpoints(tc.si.ToCreate) + countEndpoints(tc.si.ToUpdate) + countEndpoints(tc.si.Unchanged)

			if actualTotal != tc.expectedTotal {
				// This likely means that sliceWithNEndpoints or countEndpoints are not working as expected
				t.Errorf("Problem with test or test helper. Expected %d total endpoints, got %d", tc.expectedTotal, actualTotal)
			}

			actualTotalReady := tc.si.getTotalReadyEndpoints()
			if actualTotalReady != tc.expectedTotalReady {
				t.Errorf("Expected %d, got %d", tc.expectedTotalReady, actualTotalReady)
			}
		})
	}
}

// helpers

func sliceWithNEndpoints(ready, unready int) *discovery.EndpointSlice {
	endpoints := make([]discovery.Endpoint, ready+unready)

	for i := 0; i < ready; i++ {
		endpoints[i] = discovery.Endpoint{
			Addresses:  []string{fmt.Sprintf("10.1.2.%d", i)},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
		}
	}

	for i := 0; i < unready; i++ {
		endpoints[ready+i] = discovery.Endpoint{
			Addresses:  []string{fmt.Sprintf("10.1.2.%d", ready+i)},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
		}
	}

	return &discovery.EndpointSlice{
		Endpoints: endpoints,
	}
}

func countEndpoints(slices []*discovery.EndpointSlice) int {
	total := 0
	for _, slice := range slices {
		total += len(slice.Endpoints)
	}
	return total
}
