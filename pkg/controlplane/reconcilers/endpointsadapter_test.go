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

package reconcilers

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	utilnet "k8s.io/utils/net"
)

func TestEndpointsAdapterGet(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice([]int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})

	testCases := map[string]struct {
		initialState []runtime.Object

		expectedError     error
		expectedEndpoints *corev1.Endpoints
	}{
		"single-existing-endpoints": {
			initialState: []runtime.Object{endpoints1, epSlice1},

			expectedEndpoints: endpoints1,
		},
		"endpoints exists, endpointslice does not": {
			initialState: []runtime.Object{endpoints1},

			expectedEndpoints: endpoints1,
		},
		"endpointslice exists, endpoints does not": {
			initialState: []runtime.Object{epSlice1},

			expectedError: errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, testServiceName),
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1(), testServiceNamespace, testServiceName)

			endpoints, err := epAdapter.Get()

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedEndpoints) {
				t.Errorf("Wrong result from Get. Diff:\n%s", cmp.Diff(testCase.expectedEndpoints, endpoints))
			}
		})
	}
}

func TestEndpointsAdapterCreate(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice([]int{80}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice([]int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	endpointsV6, epSliceV6 := generateEndpointsAndSlice([]int{80}, []string{"1234::5678", "1234::abcd"})

	endpointsDual, _ := generateEndpointsAndSlice([]int{80}, []string{"10.1.2.3", "10.1.2.4", "1234::5678", "1234::abcd"})

	testCases := map[string]struct {
		initialState   []runtime.Object
		endpointsParam *corev1.Endpoints

		expectedError error
		expectCreate  []runtime.Object
		expectUpdate  []runtime.Object
	}{
		"single-endpoint": {
			// If the Endpoints/EndpointSlice do not exist, they will be
			// created.
			initialState:   []runtime.Object{},
			endpointsParam: endpoints1,

			expectCreate: []runtime.Object{endpoints1, epSlice1},
		},
		"single-endpoint-partial-ipv6": {
			// If the Endpoints/EndpointSlice do not exist, and the reconciler
			// erroneously tries to create a dual-stack Endpoints, we will
			// accept the erroneous Endpoints but create a single-stack IPv4
			// EndpointSlice.
			initialState:   []runtime.Object{},
			endpointsParam: endpointsDual,

			expectCreate: []runtime.Object{endpointsDual, epSlice1},
		},
		"single-endpoint-full-ipv6": {
			// If the Endpoints/EndpointSlice do not exist, and the reconciler
			// creates a single-stack IPv6 Endpoints, we will create a
			// single-stack IPv6 EndpointSlice.
			initialState:   []runtime.Object{},
			endpointsParam: endpointsV6,

			expectCreate: []runtime.Object{endpointsV6, epSliceV6},
		},
		"existing-endpoints": {
			// If the Endpoints/EndpointSlice already exist then Create will
			// return an error, because you should have called Update.
			initialState:   []runtime.Object{endpoints1, epSlice1},
			endpointsParam: endpoints1,

			expectedError: errors.NewAlreadyExists(schema.GroupResource{Group: "", Resource: "endpoints"}, testServiceName),
			// (We expect the create to be attempted, we just also expect it to fail)
			expectCreate: []runtime.Object{endpoints1},
		},
		"existing-endpointslice-correct": {
			// No error when we need to create the Endpoints but the correct
			// EndpointSlice already exists
			initialState:   []runtime.Object{epSlice1},
			endpointsParam: endpoints1,

			expectCreate: []runtime.Object{endpoints1},
		},
		"existing-endpointslice-incorrect": {
			// No error when we need to create the Endpoints but an incorrect
			// EndpointSlice already exists
			initialState:   []runtime.Object{epSlice1},
			endpointsParam: endpoints2,

			expectCreate: []runtime.Object{endpoints2},
			expectUpdate: []runtime.Object{epSlice2},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1(), testServiceNamespace, testServiceName)

			err := epAdapter.Create(testCase.endpointsParam)
			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			err = verifyActions(client, testCase.expectCreate, testCase.expectUpdate, nil)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

func TestEndpointsAdapterUpdate(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice([]int{80}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice([]int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	_, epSlice1Deprecated := generateEndpointsAndSlice([]int{80}, []string{"10.1.3", "10.1.2.4"})
	epSlice1Deprecated.AddressType = discovery.AddressType("IP")

	testCases := map[string]struct {
		initialState   []runtime.Object
		endpointsParam *corev1.Endpoints

		expectedError error
		expectCreate  []runtime.Object
		expectUpdate  []runtime.Object
		expectDelete  []runtime.Object
	}{
		"single-existing-endpoints-no-change": {
			// If the Endpoints/EndpointSlice already exist and are correct,
			// then Update will pointlessly update it anyway because you
			// shouldn't have called it if you didn't want it to do that.
			initialState:   []runtime.Object{endpoints1, epSlice1},
			endpointsParam: endpoints1,

			expectUpdate: []runtime.Object{endpoints1},
		},
		"existing-endpointslice-replaced-with-updated-ipv4-address-type": {
			// If an EndpointSlice with deprecated "IP" address type exists,
			// it is deleted and replaced with one that has "IPv4" address
			// type.
			initialState:   []runtime.Object{endpoints1, epSlice1Deprecated},
			endpointsParam: endpoints1,

			expectUpdate: []runtime.Object{endpoints1},
			expectDelete: []runtime.Object{epSlice1Deprecated},
			expectCreate: []runtime.Object{epSlice1},
		},
		"add-ports-and-ips": {
			// If we add ports/IPs to the Endpoints they will be added to
			// the EndpointSlice.
			initialState:   []runtime.Object{endpoints1, epSlice1},
			endpointsParam: endpoints2,

			expectUpdate: []runtime.Object{endpoints2, epSlice2},
		},
		"endpoints-correct-endpointslice-wrong": {
			// If the Endpoints is correct and the EndpointSlice is wrong,
			// Sync will update the EndpointSlice.
			initialState:   []runtime.Object{endpoints2, epSlice1},
			endpointsParam: endpoints2,

			expectUpdate: []runtime.Object{endpoints2, epSlice2},
		},
		"endpointslice-correct-endpoints-wrong": {
			// If the EndpointSlice is correct and the Endpoints is wrong,
			// Sync will update the Endpoints.
			initialState:   []runtime.Object{endpoints1, epSlice2},
			endpointsParam: endpoints2,

			expectUpdate: []runtime.Object{endpoints2},
		},
		"missing-endpoints": {
			// If the Endpoints/EndpointSlice doesn't already exist then
			// Update will return an error, because you should have called
			// Create.
			initialState:   []runtime.Object{},
			endpointsParam: endpoints1,

			expectedError: errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, testServiceName),
			// We expect the update to be attempted, we just also expect it to fail
			expectUpdate: []runtime.Object{endpoints1},
		},
		"missing-endpointslice": {
			// No error when we need to update the Endpoints but the
			// EndpointSlice doesn't exist
			initialState:   []runtime.Object{endpoints2},
			endpointsParam: endpoints1,

			expectUpdate: []runtime.Object{endpoints1},
			expectCreate: []runtime.Object{epSlice1},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1(), testServiceNamespace, testServiceName)

			err := epAdapter.Update(testCase.endpointsParam)
			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			err = verifyActions(client, testCase.expectCreate, testCase.expectUpdate, testCase.expectDelete)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

func generateEndpointsAndSlice(ports []int, addresses []string) (*corev1.Endpoints, *discovery.EndpointSlice) {
	trueBool := true
	addressType := discovery.AddressTypeIPv4
	if len(addresses) > 0 && utilnet.IsIPv6String(addresses[0]) {
		addressType = discovery.AddressTypeIPv6
	}

	epSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testServiceNamespace,
			Name:      testServiceName,
		},
		AddressType: addressType,
	}
	epSlice.Labels = map[string]string{discovery.LabelServiceName: testServiceName}
	subset := corev1.EndpointSubset{}

	for i, port := range ports {
		endpointPort := corev1.EndpointPort{
			Name:     fmt.Sprintf("port-%d", i),
			Port:     int32(port),
			Protocol: corev1.ProtocolTCP,
		}
		subset.Ports = append(subset.Ports, endpointPort)
		epSlice.Ports = append(epSlice.Ports, discovery.EndpointPort{
			Name:     &endpointPort.Name,
			Port:     &endpointPort.Port,
			Protocol: &endpointPort.Protocol,
		})
	}

	for _, address := range addresses {
		endpointAddress := corev1.EndpointAddress{
			IP: address,
		}

		subset.Addresses = append(subset.Addresses, endpointAddress)

		epSlice.Endpoints = append(epSlice.Endpoints, discovery.Endpoint{
			Addresses:  []string{endpointAddress.IP},
			Conditions: discovery.EndpointConditions{Ready: &trueBool},
		})
	}

	return &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testServiceName,
			Namespace: testServiceNamespace,
			Labels: map[string]string{
				discovery.LabelSkipMirror: "true",
			},
		},
		Subsets: []corev1.EndpointSubset{subset},
	}, epSlice
}

func TestEndpointsAdapterEnsureEndpointSliceFromEndpoints(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice([]int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice([]int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	testCases := map[string]struct {
		initialState   []runtime.Object
		endpointsParam *corev1.Endpoints

		expectedError         error
		expectedEndpointSlice *discovery.EndpointSlice
	}{
		"existing-endpointslice-no-change": {
			initialState:          []runtime.Object{epSlice1},
			endpointsParam:        endpoints1,
			expectedEndpointSlice: epSlice1,
		},
		"existing-endpointslice-change": {
			initialState:          []runtime.Object{epSlice1},
			endpointsParam:        endpoints2,
			expectedEndpointSlice: epSlice2,
		},
		"missing-endpointslice": {
			initialState:          []runtime.Object{},
			endpointsParam:        endpoints1,
			expectedEndpointSlice: epSlice1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1(), testServiceNamespace, testServiceName)

			err := epAdapter.EnsureEndpointSliceFromEndpoints(testCase.endpointsParam)
			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			endpointSlice, err := client.DiscoveryV1().EndpointSlices(testServiceNamespace).Get(context.TODO(), testCase.endpointsParam.Name, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				t.Fatalf("Error getting Endpoint Slice: %v", err)
			}

			if !apiequality.Semantic.DeepEqual(endpointSlice, testCase.expectedEndpointSlice) {
				t.Errorf("Wrong result after EnsureEndpointSliceFromEndpoints. Diff:\n%s", cmp.Diff(testCase.expectedEndpointSlice, endpointSlice))
			}
		})
	}
}
