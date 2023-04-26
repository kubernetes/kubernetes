/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"testing"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	netutils "k8s.io/utils/net"
)

func TestMasterCountEndpointReconciler(t *testing.T) {
	reconcileTests := []struct {
		testName          string
		ip                string
		endpointPorts     []corev1.EndpointPort
		additionalMasters int
		initialState      []runtime.Object
		expectUpdate      []runtime.Object
		expectCreate      []runtime.Object
	}{
		{
			testName:      "no existing endpoints",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  nil,
			expectCreate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints satisfy",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints satisfy, no endpointslice",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
			expectCreate: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
		},
		{
			testName:      "existing endpointslice satisfies, no endpoints",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
			expectCreate: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
		},
		{
			testName:      "existing endpoints satisfy, endpointslice is wrong",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
				makeEndpointSlice([]string{"4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
			expectUpdate: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
		},
		{
			testName:      "existing endpointslice satisfies, endpoints is wrong",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState: []runtime.Object{
				makeEndpoints([]string{"4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
				makeEndpointSlice([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
			expectUpdate: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			},
		},
		{
			testName:      "existing endpoints satisfy but too many",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"1.2.3.4", "4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters",
			ip:                "1.2.3.4",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			initialState:      makeEndpointsArray([]string{"1.2.3.4", "4.3.2.1", "4.3.2.2", "4.3.2.3", "4.3.2.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:      makeEndpointsArray([]string{"1.2.3.4", "4.3.2.2", "4.3.2.3", "4.3.2.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters + delete first",
			ip:                "4.3.2.4",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			initialState:      makeEndpointsArray([]string{"1.2.3.4", "4.3.2.1", "4.3.2.2", "4.3.2.3", "4.3.2.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:      makeEndpointsArray([]string{"4.3.2.1", "4.3.2.2", "4.3.2.3", "4.3.2.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:          "existing endpoints satisfy and endpoint addresses length less than master count",
			ip:                "4.3.2.2",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			initialState:      makeEndpointsArray([]string{"4.3.2.1", "4.3.2.2"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:      nil,
		},
		{
			testName:          "existing endpoints current IP missing and address length less than master count",
			ip:                "4.3.2.2",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			initialState:      makeEndpointsArray([]string{"4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:      makeEndpointsArray([]string{"4.3.2.1", "4.3.2.2"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints wrong IP",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints wrong port",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 9090, Protocol: "TCP"}}),
			expectUpdate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints wrong protocol",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "UDP"}}),
			expectUpdate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "existing endpoints wrong port name",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
			initialState:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName: "existing endpoints extra service ports satisfy",
			ip:       "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
				{Name: "baz", Port: 1010, Protocol: "TCP"},
			},
			initialState: makeEndpointsArray([]string{"1.2.3.4"},
				[]corev1.EndpointPort{
					{Name: "foo", Port: 8080, Protocol: "TCP"},
					{Name: "bar", Port: 1000, Protocol: "TCP"},
					{Name: "baz", Port: 1010, Protocol: "TCP"},
				},
			),
		},
		{
			testName: "existing endpoints extra service ports missing port",
			ip:       "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			initialState: makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate: makeEndpointsArray([]string{"1.2.3.4"},
				[]corev1.EndpointPort{
					{Name: "foo", Port: 8080, Protocol: "TCP"},
					{Name: "bar", Port: 1000, Protocol: "TCP"},
				},
			),
		},
		{
			testName:      "no existing sctp endpoints",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "boo", Port: 7777, Protocol: "SCTP"}},
			initialState:  nil,
			expectCreate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "boo", Port: 7777, Protocol: "SCTP"}}),
		},
	}
	for _, test := range reconcileTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(test.initialState...)
			epAdapter := NewEndpointsAdapter(fakeClient.CoreV1(), fakeClient.DiscoveryV1(),
				testServiceNamespace, testServiceName, testServiceIP, nil)
			reconciler := NewMasterCountEndpointReconciler(test.additionalMasters+1, epAdapter)
			err := reconciler.ReconcileEndpoints([]net.IP{netutils.ParseIPSloppy(test.ip)}, test.endpointPorts, true)
			if err != nil {
				t.Errorf("unexpected error reconciling: %v", err)
			}

			err = verifyActions(fakeClient, test.expectCreate, test.expectUpdate, nil)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}

	nonReconcileTests := []struct {
		testName          string
		ip                string
		endpointPorts     []corev1.EndpointPort
		additionalMasters int
		initialState      []runtime.Object
		expectUpdate      []runtime.Object
		expectCreate      []runtime.Object
	}{
		{
			testName: "existing endpoints extra service ports missing port no update",
			ip:       "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			initialState: makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate: nil,
		},
		{
			testName: "existing endpoints extra service ports, wrong ports, wrong IP",
			ip:       "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			initialState: makeEndpointsArray([]string{"4.3.2.1"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
			expectUpdate: makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
		{
			testName:      "no existing endpoints",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			initialState:  nil,
			expectCreate:  makeEndpointsArray([]string{"1.2.3.4"}, []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}),
		},
	}
	for _, test := range nonReconcileTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(test.initialState...)
			epAdapter := NewEndpointsAdapter(fakeClient.CoreV1(), fakeClient.DiscoveryV1(),
				testServiceNamespace, testServiceName, testServiceIP, nil)
			reconciler := NewMasterCountEndpointReconciler(test.additionalMasters+1, epAdapter)
			err := reconciler.ReconcileEndpoints([]net.IP{netutils.ParseIPSloppy(test.ip)}, test.endpointPorts, false)
			if err != nil {
				t.Errorf("unexpected error reconciling: %v", err)
			}

			err = verifyActions(fakeClient, test.expectCreate, test.expectUpdate, nil)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

// masterCountEndpointReconciler does not support dual-stack, so this should really only
// happen during rollouts/rollbacks or misconfiguration.
func TestMasterCountEndpointReconcilerDualStack(t *testing.T) {
	endpointPorts := []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}}

	emptyV6Slice := makeEndpointSliceSecondary([]string{"1234::5678"}, endpointPorts)
	emptyV6Slice.Endpoints = []discoveryv1.Endpoint{}

	reconcileTests := []struct {
		testName     string
		ips          []net.IP
		initialState []runtime.Object
		expectUpdate []runtime.Object
		expectCreate []runtime.Object
		expectDelete []runtime.Object
	}{
		{
			testName: "no existing endpoints (secondary IP ignored)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: nil,
			expectCreate: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
				emptyV6Slice,
			},
		},
		{
			testName: "existing endpoints satisfy (secondary IP ignored)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
				emptyV6Slice,
			},
		},
		{
			testName: "existing endpoints satisfy, no endpointslices (secondary IP ignored)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
			},
			expectCreate: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
				emptyV6Slice,
			},
		},
		{
			testName: "existing endpointslices satisfy, no endpoints (secondary slice emptied)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
				makeEndpointSliceSecondary([]string{"1234::5678"}, endpointPorts),
			},
			expectCreate: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
			},
			expectUpdate: []runtime.Object{
				emptyV6Slice,
			},
		},
		{
			testName: "existing endpoints satisfy, only v4 endpointslice (secondary IPs ignored)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
			},
			expectCreate: []runtime.Object{
				emptyV6Slice,
			},
		},
		{
			testName: "existing endpoints satisfy, only v6 endpointslice (emptied)",
			ips: []net.IP{
				netutils.ParseIPSloppy("1.2.3.4"),
				netutils.ParseIPSloppy("1234::5678"),
			},
			initialState: []runtime.Object{
				makeEndpoints([]string{"1.2.3.4"}, endpointPorts),
				makeEndpointSliceSecondary([]string{"1234::5678"}, endpointPorts),
			},
			expectCreate: []runtime.Object{
				makeEndpointSlice([]string{"1.2.3.4"}, endpointPorts),
			},
			expectUpdate: []runtime.Object{
				emptyV6Slice,
			},
		},
	}
	for _, test := range reconcileTests {
		t.Run(test.testName, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(test.initialState...)
			epAdapter := NewEndpointsAdapter(fakeClient.CoreV1(), fakeClient.DiscoveryV1(),
				testServiceNamespace, testServiceName,
				testServiceIP, testServiceIPv6)
			reconciler := NewMasterCountEndpointReconciler(3, epAdapter)
			err := reconciler.ReconcileEndpoints(test.ips, endpointPorts, true)
			if err != nil {
				t.Errorf("unexpected error reconciling: %v", err)
			}

			err = verifyActions(fakeClient, test.expectCreate, test.expectUpdate, test.expectDelete)
			if err != nil {
				t.Errorf("unexpected error in side effects: %v", err)
			}
		})
	}
}

func TestEmptySubsets(t *testing.T) {
	endpoints := makeEndpointsArray(nil, nil)
	fakeClient := fake.NewSimpleClientset(endpoints...)
	epAdapter := NewEndpointsAdapter(fakeClient.CoreV1(), fakeClient.DiscoveryV1(),
		testServiceNamespace, testServiceName, testServiceIP, nil)
	reconciler := NewMasterCountEndpointReconciler(1, epAdapter)
	endpointPorts := []corev1.EndpointPort{
		{Name: "foo", Port: 8080, Protocol: "TCP"},
	}
	err := reconciler.RemoveEndpoints([]net.IP{netutils.ParseIPSloppy("1.2.3.4")}, endpointPorts)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
