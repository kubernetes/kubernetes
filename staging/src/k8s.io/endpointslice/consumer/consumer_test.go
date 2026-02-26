/*
Copyright 2024 The Kubernetes Authors.

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

package consumer

import (
	"reflect"
	"testing"

	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestEndpointSliceConsumer(t *testing.T) {
	testCases := []struct {
		name           string
		endpointSlices []*discovery.EndpointSlice
		serviceNN      types.NamespacedName
		expectedSlices int
	}{
		{
			name: "single slice",
			endpointSlices: []*discovery.EndpointSlice{
				createTestEndpointSlice("svc1", "ns1", "slice1", []discovery.Endpoint{
					createTestEndpoint([]string{"10.0.0.1"}, ptr.To("node1"), ptr.To(true)),
					createTestEndpoint([]string{"10.0.0.2"}, ptr.To("node2"), ptr.To(true)),
				}),
			},
			serviceNN:      types.NamespacedName{Namespace: "ns1", Name: "svc1"},
			expectedSlices: 1,
		},
		{
			name: "multiple slices",
			endpointSlices: []*discovery.EndpointSlice{
				createTestEndpointSlice("svc1", "ns1", "slice1", []discovery.Endpoint{
					createTestEndpoint([]string{"10.0.0.1"}, ptr.To("node1"), ptr.To(true)),
					createTestEndpoint([]string{"10.0.0.2"}, ptr.To("node2"), ptr.To(true)),
				}),
				createTestEndpointSlice("svc1", "ns1", "slice2", []discovery.Endpoint{
					createTestEndpoint([]string{"10.0.0.3"}, ptr.To("node3"), ptr.To(true)),
					createTestEndpoint([]string{"10.0.0.4"}, ptr.To("node4"), ptr.To(true)),
				}),
			},
			serviceNN:      types.NamespacedName{Namespace: "ns1", Name: "svc1"},
			expectedSlices: 2,
		},
		{
			name: "duplicate endpoints across slices",
			endpointSlices: []*discovery.EndpointSlice{
				createTestEndpointSlice("svc1", "ns1", "slice1", []discovery.Endpoint{
					createTestEndpoint([]string{"10.0.0.1"}, ptr.To("node1"), ptr.To(true)),
					createTestEndpoint([]string{"10.0.0.2"}, ptr.To("node2"), ptr.To(true)),
				}),
				createTestEndpointSlice("svc1", "ns1", "slice2", []discovery.Endpoint{
					createTestEndpoint([]string{"10.0.0.1"}, ptr.To("node1"), ptr.To(true)), // Duplicate
					createTestEndpoint([]string{"10.0.0.3"}, ptr.To("node3"), ptr.To(true)),
				}),
			},
			serviceNN:      types.NamespacedName{Namespace: "ns1", Name: "svc1"},
			expectedSlices: 2,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			consumer := NewEndpointSliceConsumer()

			// Add all slices
			for _, slice := range tc.endpointSlices {
				consumer.OnEndpointSliceAdd(slice)
			}

			// Check GetEndpointSlices
			slices := consumer.GetEndpointSlices(tc.serviceNN)
			if len(slices) != tc.expectedSlices {
				t.Errorf("Expected %d slices, got %d", tc.expectedSlices, len(slices))
			}

			// Test event handler
			var handlerCalled bool
			var handlerServiceNN types.NamespacedName
			var handlerSlices []*discovery.EndpointSlice

			consumer.AddEventHandler(EndpointChangeHandlerFunc(func(serviceNN types.NamespacedName, slices []*discovery.EndpointSlice) {
				handlerCalled = true
				handlerServiceNN = serviceNN
				handlerSlices = slices
			}))

			// Add a new slice to trigger the handler
			newSlice := createTestEndpointSlice("svc1", "ns1", "slice3", []discovery.Endpoint{
				createTestEndpoint([]string{"10.0.0.5"}, ptr.To("node5"), ptr.To(true)),
			})
			consumer.OnEndpointSliceAdd(newSlice)

			if !handlerCalled {
				t.Error("Handler was not called")
			}
			if handlerServiceNN != tc.serviceNN {
				t.Errorf("Handler received wrong service NamespacedName: %v, expected %v", handlerServiceNN, tc.serviceNN)
			}
			if len(handlerSlices) != tc.expectedSlices+1 {
				t.Errorf("Handler received wrong number of slices: %d, expected %d", len(handlerSlices), tc.expectedSlices+1)
			}

			// Test deletion
			consumer.OnEndpointSliceDelete(newSlice)
			slices = consumer.GetEndpointSlices(tc.serviceNN)
			if len(slices) != tc.expectedSlices {
				t.Errorf("After deletion: Expected %d slices, got %d", tc.expectedSlices, len(slices))
			}
		})
	}
}

func TestEndpointSliceCacheKeys(t *testing.T) {
	testCases := []struct {
		name          string
		endpointSlice *discovery.EndpointSlice
		expectValid   bool
		expectedNN    types.NamespacedName
		expectedKey   string
	}{
		{
			name: "valid slice",
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "slice1",
					Namespace: "ns1",
					Labels: map[string]string{
						discovery.LabelServiceName: "svc1",
					},
				},
			},
			expectValid: true,
			expectedNN:  types.NamespacedName{Namespace: "ns1", Name: "svc1"},
			expectedKey: "slice1",
		},
		{
			name: "missing service name label",
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "slice1",
					Namespace: "ns1",
				},
			},
			expectValid: false,
		},
		{
			name: "empty service name label",
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "slice1",
					Namespace: "ns1",
					Labels: map[string]string{
						discovery.LabelServiceName: "",
					},
				},
			},
			expectValid: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nn, key, ok := endpointSliceCacheKeys(tc.endpointSlice)
			if ok != tc.expectValid {
				t.Errorf("Expected valid=%v, got valid=%v", tc.expectValid, ok)
			}
			if tc.expectValid {
				if !reflect.DeepEqual(nn, tc.expectedNN) {
					t.Errorf("Expected NamespacedName %v, got %v", tc.expectedNN, nn)
				}
				if key != tc.expectedKey {
					t.Errorf("Expected key %s, got %s", tc.expectedKey, key)
				}
			}
		})
	}
}

// Helper functions to create test objects

func createTestEndpointSlice(serviceName, namespace, name string, endpoints []discovery.Endpoint) *discovery.EndpointSlice {
	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints:   endpoints,
	}
}

func createTestEndpoint(addresses []string, nodeName *string, ready *bool) discovery.Endpoint {
	return discovery.Endpoint{
		Addresses: addresses,
		NodeName:  nodeName,
		Conditions: discovery.EndpointConditions{
			Ready: ready,
		},
	}
}
