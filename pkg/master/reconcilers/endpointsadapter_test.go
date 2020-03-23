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
	discovery "k8s.io/api/discovery/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
)

func TestEndpointsAdapterGet(t *testing.T) {
	endpoints1, _ := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})

	testCases := map[string]struct {
		endpointSlicesEnabled bool
		expectedError         error
		expectedEndpoints     *corev1.Endpoints
		endpoints             []*corev1.Endpoints
		namespaceParam        string
		nameParam             string
	}{
		"single-existing-endpoints": {
			endpointSlicesEnabled: false,
			expectedError:         nil,
			expectedEndpoints:     endpoints1,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			nameParam:             "foo",
		},
		"single-existing-endpoints-slices-enabled": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints1,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			nameParam:             "foo",
		},
		"wrong-namespace": {
			endpointSlicesEnabled: false,
			expectedError:         errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "foo"),
			expectedEndpoints:     nil,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "foo",
			nameParam:             "foo",
		},
		"wrong-name": {
			endpointSlicesEnabled: false,
			expectedError:         errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "bar"),
			expectedEndpoints:     nil,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			nameParam:             "bar",
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			epAdapter := EndpointsAdapter{endpointClient: client.CoreV1()}
			if testCase.endpointSlicesEnabled {
				epAdapter.endpointSliceClient = client.DiscoveryV1beta1()
			}

			for _, endpoints := range testCase.endpoints {
				_, err := client.CoreV1().Endpoints(endpoints.Namespace).Create(context.TODO(), endpoints, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating Endpoints: %v", err)
				}
			}

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
		endpointSlicesEnabled bool
		expectedError         error
		expectedEndpoints     *corev1.Endpoints
		expectedEndpointSlice *discovery.EndpointSlice
		endpoints             []*corev1.Endpoints
		endpointSlices        []*discovery.EndpointSlice
		namespaceParam        string
		endpointsParam        *corev1.Endpoints
	}{
		"single-endpoint": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints1,
			expectedEndpointSlice: epSlice1,
			endpoints:             []*corev1.Endpoints{},
			namespaceParam:        endpoints1.Namespace,
			endpointsParam:        endpoints1,
		},
		"single-endpoint-partial-ipv6": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints2,
			expectedEndpointSlice: epSlice2,
			endpoints:             []*corev1.Endpoints{},
			namespaceParam:        endpoints2.Namespace,
			endpointsParam:        endpoints2,
		},
		"single-endpoint-full-ipv6": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints3,
			expectedEndpointSlice: epSlice3,
			endpoints:             []*corev1.Endpoints{},
			namespaceParam:        endpoints3.Namespace,
			endpointsParam:        endpoints3,
		},
		"single-endpoint-no-slices": {
			endpointSlicesEnabled: false,
			expectedError:         nil,
			expectedEndpoints:     endpoints1,
			expectedEndpointSlice: nil,
			endpoints:             []*corev1.Endpoints{},
			namespaceParam:        endpoints1.Namespace,
			endpointsParam:        endpoints1,
		},
		"existing-endpoint": {
			endpointSlicesEnabled: true,
			expectedError:         errors.NewAlreadyExists(schema.GroupResource{Group: "", Resource: "endpoints"}, "foo"),
			expectedEndpoints:     nil,
			expectedEndpointSlice: nil,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        endpoints1.Namespace,
			endpointsParam:        endpoints1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			epAdapter := EndpointsAdapter{endpointClient: client.CoreV1()}
			if testCase.endpointSlicesEnabled {
				epAdapter.endpointSliceClient = client.DiscoveryV1beta1()
			}

			for _, endpoints := range testCase.endpoints {
				_, err := client.CoreV1().Endpoints(endpoints.Namespace).Create(context.TODO(), endpoints, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating Endpoints: %v", err)
				}
			}

			endpoints, err := epAdapter.Create(testCase.namespaceParam, testCase.endpointsParam)

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedEndpoints) {
				t.Errorf("Expected endpoints: %v, got: %v", testCase.expectedEndpoints, endpoints)
			}

			epSliceList, err := client.DiscoveryV1beta1().EndpointSlices(testCase.namespaceParam).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				t.Fatalf("Error listing Endpoint Slices: %v", err)
			}

			if testCase.expectedEndpointSlice == nil {
				if len(epSliceList.Items) != 0 {
					t.Fatalf("Expected no Endpoint Slices, got: %v", epSliceList.Items)
				}
			} else {
				if len(epSliceList.Items) == 0 {
					t.Fatalf("No Endpoint Slices found, expected: %v", testCase.expectedEndpointSlice)
				}
				if len(epSliceList.Items) > 1 {
					t.Errorf("Only 1 Endpoint Slice expected, got: %v", testCase.expectedEndpointSlice)
				}
				if !apiequality.Semantic.DeepEqual(*testCase.expectedEndpointSlice, epSliceList.Items[0]) {
					t.Errorf("Expected Endpoint Slice: %v, got: %v", testCase.expectedEndpointSlice, epSliceList.Items[0])

				}
			}
		})
	}
}

func TestEndpointsAdapterUpdate(t *testing.T) {
	endpoints1, _ := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})
	endpoints3, _ := generateEndpointsAndSlice("bar", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	// ensure that EndpointSlice with deprecated IP address type is replaced
	// with one that has an IPv4 address type.
	endpoints4, _ := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})
	_, epSlice4IP := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})
	epSlice4IP.AddressType = discovery.AddressTypeIP
	_, epSlice4IPv4 := generateEndpointsAndSlice("foo", "testing", []int{80}, []string{"10.1.2.7", "10.1.2.8"})

	testCases := map[string]struct {
		endpointSlicesEnabled bool
		expectedError         error
		expectedEndpoints     *corev1.Endpoints
		expectedEndpointSlice *discovery.EndpointSlice
		endpoints             []*corev1.Endpoints
		endpointSlices        []*discovery.EndpointSlice
		namespaceParam        string
		endpointsParam        *corev1.Endpoints
	}{
		"single-existing-endpoints-no-change": {
			endpointSlicesEnabled: false,
			expectedError:         nil,
			expectedEndpoints:     endpoints1,
			expectedEndpointSlice: nil,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
		"existing-endpointslice-replaced-with-updated-ipv4-address-type": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints4,
			expectedEndpointSlice: epSlice4IPv4,
			endpoints:             []*corev1.Endpoints{endpoints4},
			endpointSlices:        []*discovery.EndpointSlice{epSlice4IP},
			namespaceParam:        "testing",
			endpointsParam:        endpoints4,
		},
		"add-ports-and-ips": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpoints:     endpoints2,
			expectedEndpointSlice: epSlice2,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints2,
		},
		"missing-endpoints": {
			endpointSlicesEnabled: true,
			expectedError:         errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "bar"),
			expectedEndpoints:     nil,
			expectedEndpointSlice: nil,
			endpoints:             []*corev1.Endpoints{endpoints1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints3,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			epAdapter := EndpointsAdapter{endpointClient: client.CoreV1()}
			if testCase.endpointSlicesEnabled {
				epAdapter.endpointSliceClient = client.DiscoveryV1beta1()
			}

			for _, endpoints := range testCase.endpoints {
				_, err := client.CoreV1().Endpoints(endpoints.Namespace).Create(context.TODO(), endpoints, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating Endpoints: %v", err)
				}
			}

			endpoints, err := epAdapter.Update(testCase.namespaceParam, testCase.endpointsParam)

			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			if !apiequality.Semantic.DeepEqual(endpoints, testCase.expectedEndpoints) {
				t.Errorf("Expected endpoints: %v, got: %v", testCase.expectedEndpoints, endpoints)
			}

			epSliceList, err := client.DiscoveryV1beta1().EndpointSlices(testCase.namespaceParam).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				t.Fatalf("Error listing Endpoint Slices: %v", err)
			}

			if testCase.expectedEndpointSlice == nil {
				if len(epSliceList.Items) != 0 {
					t.Fatalf("Expected no Endpoint Slices, got: %v", epSliceList.Items)
				}
			} else {
				if len(epSliceList.Items) == 0 {
					t.Fatalf("No Endpoint Slices found, expected: %v", testCase.expectedEndpointSlice)
				}
				if len(epSliceList.Items) > 1 {
					t.Errorf("Only 1 Endpoint Slice expected, got: %v", testCase.expectedEndpointSlice)
				}
				if !apiequality.Semantic.DeepEqual(*testCase.expectedEndpointSlice, epSliceList.Items[0]) {
					t.Errorf("Expected Endpoint Slice: %v, got: %v", testCase.expectedEndpointSlice, epSliceList.Items[0])

				}
			}
		})
	}
}

func generateEndpointsAndSlice(name, namespace string, ports []int, addresses []string) (*corev1.Endpoints, *discovery.EndpointSlice) {
	objectMeta := metav1.ObjectMeta{Name: name, Namespace: namespace}
	trueBool := true

	epSlice := &discovery.EndpointSlice{ObjectMeta: objectMeta, AddressType: discovery.AddressTypeIPv4}
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
		ObjectMeta: objectMeta,
		Subsets:    []corev1.EndpointSubset{subset},
	}, epSlice
}

func TestEndpointsAdapterEnsureEndpointSliceFromEndpoints(t *testing.T) {
	endpoints1, epSlice1 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4"})
	endpoints2, epSlice2 := generateEndpointsAndSlice("foo", "testing", []int{80, 443}, []string{"10.1.2.3", "10.1.2.4", "10.1.2.5"})

	testCases := map[string]struct {
		endpointSlicesEnabled bool
		expectedError         error
		expectedEndpointSlice *discovery.EndpointSlice
		endpointSlices        []*discovery.EndpointSlice
		namespaceParam        string
		endpointsParam        *corev1.Endpoints
	}{
		"existing-endpointslice-no-change": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpointSlice: epSlice1,
			endpointSlices:        []*discovery.EndpointSlice{epSlice1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
		"existing-endpointslice-change": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpointSlice: epSlice2,
			endpointSlices:        []*discovery.EndpointSlice{epSlice1},
			namespaceParam:        "testing",
			endpointsParam:        endpoints2,
		},
		"missing-endpointslice": {
			endpointSlicesEnabled: true,
			expectedError:         nil,
			expectedEndpointSlice: epSlice1,
			endpointSlices:        []*discovery.EndpointSlice{},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
		"endpointslices-disabled": {
			endpointSlicesEnabled: false,
			expectedError:         nil,
			expectedEndpointSlice: nil,
			endpointSlices:        []*discovery.EndpointSlice{},
			namespaceParam:        "testing",
			endpointsParam:        endpoints1,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			epAdapter := EndpointsAdapter{endpointClient: client.CoreV1()}
			if testCase.endpointSlicesEnabled {
				epAdapter.endpointSliceClient = client.DiscoveryV1beta1()
			}

			for _, endpointSlice := range testCase.endpointSlices {
				_, err := client.DiscoveryV1beta1().EndpointSlices(endpointSlice.Namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Error creating EndpointSlice: %v", err)
				}
			}

			err := epAdapter.EnsureEndpointSliceFromEndpoints(testCase.namespaceParam, testCase.endpointsParam)
			if !apiequality.Semantic.DeepEqual(testCase.expectedError, err) {
				t.Errorf("Expected error: %v, got: %v", testCase.expectedError, err)
			}

			endpointSlice, err := client.DiscoveryV1beta1().EndpointSlices(testCase.namespaceParam).Get(context.TODO(), testCase.endpointsParam.Name, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				t.Fatalf("Error getting Endpoint Slice: %v", err)
			}

			if !apiequality.Semantic.DeepEqual(endpointSlice, testCase.expectedEndpointSlice) {
				t.Errorf("Expected Endpoint Slice: %v, got: %v", testCase.expectedEndpointSlice, endpointSlice)
			}
		})
	}
}
