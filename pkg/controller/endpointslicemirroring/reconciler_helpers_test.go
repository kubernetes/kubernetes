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
	"sort"
	"testing"

	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRecycleSlices(t *testing.T) {
	testCases := []struct {
		testName       string
		startingSlices *slicesByAction
		expectedSlices *slicesByAction
	}{{
		testName:       "Empty slices",
		startingSlices: &slicesByAction{},
		expectedSlices: &slicesByAction{},
	}, {
		testName: "1 to create and 1 to delete",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{simpleEndpointSlice("foo", "10.1.2.3", discovery.AddressTypeIPv4)},
			toDelete: []*discovery.EndpointSlice{simpleEndpointSlice("bar", "10.2.3.4", discovery.AddressTypeIPv4)},
		},
		expectedSlices: &slicesByAction{
			toUpdate: []*discovery.EndpointSlice{simpleEndpointSlice("bar", "10.1.2.3", discovery.AddressTypeIPv4)},
		},
	}, {
		testName: "1 to create, update, and delete",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{simpleEndpointSlice("foo", "10.1.2.3", discovery.AddressTypeIPv4)},
			toUpdate: []*discovery.EndpointSlice{simpleEndpointSlice("baz", "10.2.3.4", discovery.AddressTypeIPv4)},
			toDelete: []*discovery.EndpointSlice{simpleEndpointSlice("bar", "10.3.4.5", discovery.AddressTypeIPv4)},
		},
		expectedSlices: &slicesByAction{
			toUpdate: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar", "10.1.2.3", discovery.AddressTypeIPv4),
				simpleEndpointSlice("baz", "10.2.3.4", discovery.AddressTypeIPv4),
			},
		},
	}, {
		testName: "2 to create and 1 to delete",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{
				simpleEndpointSlice("foo1", "10.1.2.3", discovery.AddressTypeIPv4),
				simpleEndpointSlice("foo2", "10.3.4.5", discovery.AddressTypeIPv4),
			},
			toDelete: []*discovery.EndpointSlice{simpleEndpointSlice("bar", "10.2.3.4", discovery.AddressTypeIPv4)},
		},
		expectedSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{simpleEndpointSlice("foo2", "10.3.4.5", discovery.AddressTypeIPv4)},
			toUpdate: []*discovery.EndpointSlice{simpleEndpointSlice("bar", "10.1.2.3", discovery.AddressTypeIPv4)},
		},
	}, {
		testName: "1 to create and 2 to delete",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{
				simpleEndpointSlice("foo1", "10.1.2.3", discovery.AddressTypeIPv4),
			},
			toDelete: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar1", "10.2.3.4", discovery.AddressTypeIPv4),
				simpleEndpointSlice("bar2", "10.3.4.5", discovery.AddressTypeIPv4),
			},
		},
		expectedSlices: &slicesByAction{
			toUpdate: []*discovery.EndpointSlice{simpleEndpointSlice("bar1", "10.1.2.3", discovery.AddressTypeIPv4)},
			toDelete: []*discovery.EndpointSlice{simpleEndpointSlice("bar2", "10.3.4.5", discovery.AddressTypeIPv4)},
		},
	}, {
		testName: "1 to create and 1 to delete for each IP family",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{
				simpleEndpointSlice("foo-v4", "10.1.2.3", discovery.AddressTypeIPv4),
				simpleEndpointSlice("foo-v6", "2001:db8:1111:3333:4444:5555:6666:7777", discovery.AddressTypeIPv6),
			},
			toDelete: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar-v4", "10.2.2.3", discovery.AddressTypeIPv4),
				simpleEndpointSlice("bar-v6", "2001:db8:2222:3333:4444:5555:6666:7777", discovery.AddressTypeIPv6),
			},
		},
		expectedSlices: &slicesByAction{
			toUpdate: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar-v4", "10.1.2.3", discovery.AddressTypeIPv4),
				simpleEndpointSlice("bar-v6", "2001:db8:1111:3333:4444:5555:6666:7777", discovery.AddressTypeIPv6),
			},
		},
	}, {
		testName: "1 to create and 1 to delete, wrong IP family",
		startingSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{
				simpleEndpointSlice("foo-v4", "10.1.2.3", discovery.AddressTypeIPv4),
			},
			toDelete: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar-v6", "2001:db8:2222:3333:4444:5555:6666:7777", discovery.AddressTypeIPv6),
			},
		},
		expectedSlices: &slicesByAction{
			toCreate: []*discovery.EndpointSlice{
				simpleEndpointSlice("foo-v4", "10.1.2.3", discovery.AddressTypeIPv4),
			},
			toDelete: []*discovery.EndpointSlice{
				simpleEndpointSlice("bar-v6", "2001:db8:2222:3333:4444:5555:6666:7777", discovery.AddressTypeIPv6),
			},
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			startingSlices := tc.startingSlices
			recycleSlices(startingSlices)

			unorderedSlices := [][]*discovery.EndpointSlice{startingSlices.toCreate, startingSlices.toUpdate, startingSlices.toDelete}
			for _, actual := range unorderedSlices {
				sort.Slice(actual, func(i, j int) bool {
					return actual[i].Name < actual[j].Name
				})
			}

			expectEqualSlices(t, startingSlices.toCreate, tc.expectedSlices.toCreate)
			expectEqualSlices(t, startingSlices.toUpdate, tc.expectedSlices.toUpdate)
			expectEqualSlices(t, startingSlices.toDelete, tc.expectedSlices.toDelete)
		})
	}
}

// Test helpers
func expectEqualSlices(t *testing.T, actual, expected []*discovery.EndpointSlice) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("Expected %d EndpointSlices, got %d: %v", len(expected), len(actual), actual)
	}

	for i, expectedSlice := range expected {
		if expectedSlice.AddressType != actual[i].AddressType {
			t.Errorf("Expected Slice to have %s address type, got %s", expectedSlice.AddressType, actual[i].AddressType)
		}

		if expectedSlice.Name != actual[i].Name {
			t.Errorf("Expected Slice to have %s name, got %s", expectedSlice.Name, actual[i].Name)
		}

		if len(expectedSlice.Endpoints) != len(actual[i].Endpoints) {
			t.Fatalf("Expected Slice to have %d endpoints, got %d", len(expectedSlice.Endpoints), len(actual[i].Endpoints))
		}

		for j, expectedEndpoint := range expectedSlice.Endpoints {
			actualEndpoint := actual[i].Endpoints[j]
			if len(expectedEndpoint.Addresses) != len(actualEndpoint.Addresses) {
				t.Fatalf("Expected Endpoint to have %d addresses, got %d", len(expectedEndpoint.Addresses), len(actualEndpoint.Addresses))
			}

			for k, expectedAddress := range expectedEndpoint.Addresses {
				actualAddress := actualEndpoint.Addresses[k]
				if expectedAddress != actualAddress {
					t.Fatalf("Expected address to be %s, got %s", expectedAddress, actualAddress)
				}
			}
		}
	}
}

func simpleEndpointSlice(name, ip string, addrType discovery.AddressType) *discovery.EndpointSlice {
	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		AddressType: addrType,
		Endpoints: []discovery.Endpoint{{
			Addresses: []string{ip},
		}},
	}
}
