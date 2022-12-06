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

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
)

func TestEndpointsAdapterGet(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})

	testCases := map[string]struct {
		expectedError     error
		expectedEndpoints *corev1.Endpoints
		initialState      []runtime.Object
		namespaceParam    string
		nameParam         string
	}{
		"single-existing-endpoints": {
			expectedError:     nil,
			expectedEndpoints: endpoints1,
			initialState:      []runtime.Object{endpoints1, epSlice1},
			namespaceParam:    "testing",
			nameParam:         "foo",
		},
		"endpoints exists, endpointslice does not": {
			expectedError:     nil,
			expectedEndpoints: endpoints1,
			initialState:      []runtime.Object{endpoints1},
			namespaceParam:    "testing",
			nameParam:         "foo",
		},
		"endpointslice exists, endpoints does not": {
			expectedError:     errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "foo"),
			expectedEndpoints: nil,
			initialState:      []runtime.Object{epSlice1},
			namespaceParam:    "testing",
			nameParam:         "foo",
		},
		"wrong-namespace": {
			expectedError:     errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "foo"),
			expectedEndpoints: nil,
			initialState:      []runtime.Object{endpoints1, epSlice1},
			namespaceParam:    "foo",
			nameParam:         "foo",
		},
		"wrong-name": {
			expectedError:     errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "bar"),
			expectedEndpoints: nil,
			initialState:      []runtime.Object{endpoints1, epSlice1},
			namespaceParam:    "testing",
			nameParam:         "bar",
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1())

			endpoints, err := epAdapter.Get(testCase.namespaceParam, testCase.nameParam, metav1.GetOptions{})

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedEndpoints) {
				t.Errorf("Expected endpoints: %v, got: %v", testCase.expectedEndpoints, endpoints)
			}
		})
	}
}

func TestEndpointsAdapterCreate(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.3", "10.1.2.4"})

	// even if an Endpoints resource includes an IPv6 address, it should not be
	// included in the corresponding EndpointSlice.
	endpoints2, _ := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.5", "10.1.2.6", "1234::5678:0000:0000:9abc:def0"})
	_, epSlice2 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.5", "10.1.2.6"})

	// ensure that Endpoints with only IPv6 addresses result in EndpointSlice
	// with an IPv6 address type.
	endpoints3, epSlice3 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"1234::5678:0000:0000:9abc:def0"})
	epSlice3.AddressType = discovery.AddressTypeIPv6

	testCases := map[string]struct {
		expectedError  error
		expectedResult *corev1.Endpoints
		expectCreate   []runtime.Object
		expectUpdate   []runtime.Object
		initialState   []runtime.Object
		namespaceParam string
		endpointsParam *corev1.Endpoints
	}{
		"single-endpoint": {
			expectedError:  nil,
			expectedResult: endpoints1,
			expectCreate:   []runtime.Object{endpoints1, epSlice1},
			initialState:   []runtime.Object{},
			namespaceParam: endpoints1.Namespace,
			endpointsParam: endpoints1,
		},
		"single-endpoint-partial-ipv6": {
			expectedError:  nil,
			expectedResult: endpoints2,
			expectCreate:   []runtime.Object{endpoints2, epSlice2},
			initialState:   []runtime.Object{},
			namespaceParam: endpoints2.Namespace,
			endpointsParam: endpoints2,
		},
		"single-endpoint-full-ipv6": {
			expectedError:  nil,
			expectedResult: endpoints3,
			expectCreate:   []runtime.Object{endpoints3, epSlice3},
			initialState:   []runtime.Object{},
			namespaceParam: endpoints3.Namespace,
			endpointsParam: endpoints3,
		},
		"existing-endpoints": {
			expectedError:  errors.NewAlreadyExists(schema.GroupResource{Group: "", Resource: "endpoints"}, "foo"),
			expectedResult: nil,
			initialState:   []runtime.Object{endpoints1, epSlice1},
			namespaceParam: endpoints1.Namespace,
			endpointsParam: endpoints1,

			// We expect the create to be attempted, we just also expect it to fail
			expectCreate: []runtime.Object{endpoints1},
		},
		"existing-endpointslice-incorrect": {
			// No error when we need to create the Endpoints but the correct
			// EndpointSlice already exists
			expectedError:  nil,
			expectedResult: endpoints1,
			expectCreate:   []runtime.Object{endpoints1},
			initialState:   []runtime.Object{epSlice1},
			namespaceParam: endpoints1.Namespace,
			endpointsParam: endpoints1,
		},
		"existing-endpointslice-correct": {
			// No error when we need to create the Endpoints but an incorrect
			// EndpointSlice already exists
			expectedError:  nil,
			expectedResult: endpoints2,
			expectCreate:   []runtime.Object{endpoints2},
			expectUpdate:   []runtime.Object{epSlice2},
			initialState:   []runtime.Object{epSlice1},
			namespaceParam: endpoints2.Namespace,
			endpointsParam: endpoints2,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1())

			endpoints, err := epAdapter.Create(testCase.namespaceParam, testCase.endpointsParam)

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedResult) {
				t.Errorf("Expected endpoints: %v, got: %v", testCase.expectedResult, endpoints)
			}

			err = verifyCreatesAndUpdates(client, testCase.expectCreate, testCase.expectUpdate)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

func TestEndpointsAdapterUpdate(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})
	endpoints3, _ := generateEndpointsAndSlice("bar", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	// ensure that EndpointSlice with deprecated IP address type is replaced
	// with one that has an IPv4 address type.
	endpoints4, _ := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})
	_, epSlice4IP := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})
	// "IP" is a deprecated address type, ensuring that it is handled properly.
	epSlice4IP.AddressType = discovery.AddressType("IP")
	_, epSlice4IPv4 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})

	testCases := map[string]struct {
		expectedError  error
		expectedResult *corev1.Endpoints
		expectCreate   []runtime.Object
		expectUpdate   []runtime.Object
		initialState   []runtime.Object
		namespaceParam string
		endpointsParam *corev1.Endpoints
	}{
		"single-existing-endpoints-no-change": {
			expectedError:  nil,
			expectedResult: endpoints1,
			initialState:   []runtime.Object{endpoints1, epSlice1},
			namespaceParam: "testing",
			endpointsParam: endpoints1,

			// Even though there's no change, we still expect Update() to be
			// called, because this unit test ALWAYS calls Update().
			expectUpdate: []runtime.Object{endpoints1},
		},
		"existing-endpointslice-replaced-with-updated-ipv4-address-type": {
			expectedError:  nil,
			expectedResult: endpoints4,
			initialState:   []runtime.Object{endpoints4, epSlice4IP},
			namespaceParam: "testing",
			endpointsParam: endpoints4,

			// When AddressType changes, we Delete+Create the EndpointSlice,
			// so that shows up in expectCreate, not expectUpdate.
			expectUpdate: []runtime.Object{endpoints4},
			expectCreate: []runtime.Object{epSlice4IPv4},
		},
		"add-ports-and-ips": {
			expectedError:  nil,
			expectedResult: endpoints2,
			expectUpdate:   []runtime.Object{endpoints2, epSlice2},
			initialState:   []runtime.Object{endpoints1, epSlice1},
			namespaceParam: "testing",
			endpointsParam: endpoints2,
		},
		"endpoints-correct-endpointslice-wrong": {
			expectedError:  nil,
			expectedResult: endpoints2,
			expectUpdate:   []runtime.Object{endpoints2, epSlice2},
			initialState:   []runtime.Object{endpoints2, epSlice1},
			namespaceParam: "testing",
			endpointsParam: endpoints2,
		},
		"endpointslice-correct-endpoints-wrong": {
			expectedError:  nil,
			expectedResult: endpoints2,
			expectUpdate:   []runtime.Object{endpoints2},
			initialState:   []runtime.Object{endpoints1, epSlice2},
			namespaceParam: "testing",
			endpointsParam: endpoints2,
		},
		"wrong-endpoints": {
			expectedError:  errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "bar"),
			expectedResult: nil,
			expectUpdate:   []runtime.Object{endpoints3},
			initialState:   []runtime.Object{endpoints1, epSlice1},
			namespaceParam: "testing",
			endpointsParam: endpoints3,
		},
		"missing-endpoints": {
			expectedError:  errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "bar"),
			expectedResult: nil,
			initialState:   []runtime.Object{endpoints1, epSlice1},
			namespaceParam: "testing",
			endpointsParam: endpoints3,

			// We expect the update to be attempted, we just also expect it to fail
			expectUpdate: []runtime.Object{endpoints3},
		},
		"missing-endpointslice": {
			// No error when we need to update the Endpoints but the
			// EndpointSlice doesn't exist
			expectedError:  nil,
			expectedResult: endpoints1,
			expectUpdate:   []runtime.Object{endpoints1},
			expectCreate:   []runtime.Object{epSlice1},
			initialState:   []runtime.Object{endpoints2},
			namespaceParam: "testing",
			endpointsParam: endpoints1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1())

			endpoints, err := epAdapter.Update(testCase.namespaceParam, testCase.endpointsParam)

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedResult) {
				t.Errorf("Expected endpoints: %v, got: %v", testCase.expectedResult, endpoints)
			}

			err = verifyCreatesAndUpdates(client, testCase.expectCreate, testCase.expectUpdate)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

func generateEndpointsAndSlice(name, namespace string, ports []int, addresses []string) (*corev1.Endpoints, *discovery.EndpointSlice) {
	trueBool := true

	epSlice := &discovery.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Name: name, Namespace: namespace},
		AddressType: discovery.AddressTypeIPv4,
	}
	epSlice.Labels = map[string]string{discovery.LabelServiceName: name}
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

	for i, address := range addresses {
		endpointAddress := corev1.EndpointAddress{
			IP: address,
			TargetRef: &corev1.ObjectReference{
				Kind: "Pod",
				Name: fmt.Sprintf("pod-%d", i),
			},
		}

		subset.Addresses = append(subset.Addresses, endpointAddress)

		epSlice.Endpoints = append(epSlice.Endpoints, discovery.Endpoint{
			Addresses:  []string{endpointAddress.IP},
			TargetRef:  endpointAddress.TargetRef,
			Conditions: discovery.EndpointConditions{Ready: &trueBool},
		})
	}

	return &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				discovery.LabelSkipMirror: "true",
			},
		},
		Subsets: []corev1.EndpointSubset{subset},
	}, epSlice
}

func TestEndpointManagerEnsureEndpointSliceFromEndpoints(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	testCases := map[string]struct {
		expectedError         error
		expectedEndpointSlice *discovery.EndpointSlice
		initialState          []runtime.Object
		namespaceParam        string
		endpointsParam        *corev1.Endpoints
	}{
		"existing-endpointslice-no-change": {
			expectedError:         nil,
			expectedEndpointSlice: epSlice1,
			initialState:          []runtime.Object{epSlice1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
		"existing-endpointslice-change": {
			expectedError:         nil,
			expectedEndpointSlice: epSlice2,
			initialState:          []runtime.Object{epSlice1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints2,
		},
		"missing-endpointslice": {
			expectedError:         nil,
			expectedEndpointSlice: epSlice1,
			initialState:          []runtime.Object{},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset(testCase.initialState...)
			epAdapter := NewEndpointsAdapter(client.CoreV1(), client.DiscoveryV1())

			err := epAdapter.EnsureEndpointSliceFromEndpoints(testCase.namespaceParam, testCase.endpointsParam)
			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			endpointSlice, err := client.DiscoveryV1().EndpointSlices(testCase.namespaceParam).Get(context.TODO(), testCase.endpointsParam.Name, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				t.Fatalf("Error getting Endpoint Slice: %v", err)
			}

			if !apiequality.Semantic.DeepEqual(endpointSlice, testCase.expectedEndpointSlice) {
				t.Errorf("Expected Endpoint Slice: %v, got: %v", testCase.expectedEndpointSlice, endpointSlice)
			}
		})
	}
}
