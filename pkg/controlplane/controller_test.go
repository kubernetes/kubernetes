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

package controlplane

import (
	"net"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
)

func TestReconcileEndpoints(t *testing.T) {
	ns := metav1.NamespaceDefault
	om := func(name string, skipMirrorLabel bool) metav1.ObjectMeta {
		o := metav1.ObjectMeta{Namespace: ns, Name: name}
		if skipMirrorLabel {
			o.Labels = map[string]string{
				discoveryv1beta1.LabelSkipMirror: "true",
			}
		}
		return o
	}
	reconcileTests := []struct {
		testName          string
		serviceName       string
		ip                string
		endpointPorts     []corev1.EndpointPort
		additionalMasters int
		endpoints         *corev1.EndpointsList
		expectUpdate      *corev1.Endpoints // nil means none expected
		expectCreate      *corev1.Endpoints // nil means none expected
	}{
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectCreate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy but too many",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters",
			serviceName:       "foo",
			ip:                "1.2.3.4",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "1.2.3.4"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters + delete first",
			serviceName:       "foo",
			ip:                "4.3.2.4",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy and endpoint addresses length less than master count",
			serviceName:       "foo",
			ip:                "4.3.2.2",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: nil,
		},
		{
			testName:          "existing endpoints current IP missing and address length less than master count",
			serviceName:       "foo",
			ip:                "4.3.2.2",
			endpointPorts:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{
							{IP: "4.3.2.1"},
						},
						Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
					},
					Ports: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("bar", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectCreate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong IP",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 9090, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong protocol",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "UDP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints extra service ports satisfy",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
				{Name: "baz", Port: 1010, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports: []corev1.EndpointPort{
							{Name: "foo", Port: 8080, Protocol: "TCP"},
							{Name: "bar", Port: 1000, Protocol: "TCP"},
							{Name: "baz", Port: 1010, Protocol: "TCP"},
						},
					}},
				}},
			},
		},
		{
			testName:    "existing endpoints extra service ports missing port",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports: []corev1.EndpointPort{
						{Name: "foo", Port: 8080, Protocol: "TCP"},
						{Name: "bar", Port: 1000, Protocol: "TCP"},
					},
				}},
			},
		},
		{
			testName:      "no existing sctp endpoints",
			serviceName:   "boo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "boo", Port: 7777, Protocol: "SCTP"}},
			endpoints:     nil,
			expectCreate: &corev1.Endpoints{
				ObjectMeta: om("boo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "boo", Port: 7777, Protocol: "SCTP"}},
				}},
			},
		},
	}
	for _, test := range reconcileTests {
		fakeClient := fake.NewSimpleClientset()
		if test.endpoints != nil {
			fakeClient = fake.NewSimpleClientset(test.endpoints)
		}
		epAdapter := reconcilers.NewEndpointsAdapter(fakeClient.CoreV1(), nil)
		reconciler := reconcilers.NewMasterCountEndpointReconciler(test.additionalMasters+1, epAdapter)
		err := reconciler.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, true)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}

		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() != "update" {
				continue
			}
			updates = append(updates, action.(core.UpdateAction))
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else if e, a := test.expectUpdate, updates[0].GetObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}

		creates := []core.CreateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() != "create" {
				continue
			}
			creates = append(creates, action.(core.CreateAction))
		}
		if test.expectCreate != nil {
			if len(creates) != 1 {
				t.Errorf("case %q: unexpected creates: %v", test.testName, creates)
			} else if e, a := test.expectCreate, creates[0].GetObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectCreate == nil && len(creates) > 0 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
		}

	}

	nonReconcileTests := []struct {
		testName          string
		serviceName       string
		ip                string
		endpointPorts     []corev1.EndpointPort
		additionalMasters int
		endpoints         *corev1.EndpointsList
		expectUpdate      *corev1.Endpoints // nil means none expected
		expectCreate      *corev1.Endpoints // nil means none expected
	}{
		{
			testName:    "existing endpoints extra service ports missing port no update",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: nil,
		},
		{
			testName:    "existing endpoints extra service ports, wrong ports, wrong IP",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &corev1.EndpointsList{
				Items: []corev1.Endpoints{{
					ObjectMeta: om("foo", true),
					Subsets: []corev1.EndpointSubset{{
						Addresses: []corev1.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectCreate: &corev1.Endpoints{
				ObjectMeta: om("foo", true),
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []corev1.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
	}
	for _, test := range nonReconcileTests {
		fakeClient := fake.NewSimpleClientset()
		if test.endpoints != nil {
			fakeClient = fake.NewSimpleClientset(test.endpoints)
		}
		epAdapter := reconcilers.NewEndpointsAdapter(fakeClient.CoreV1(), nil)
		reconciler := reconcilers.NewMasterCountEndpointReconciler(test.additionalMasters+1, epAdapter)
		err := reconciler.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, false)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}

		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() != "update" {
				continue
			}
			updates = append(updates, action.(core.UpdateAction))
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else if e, a := test.expectUpdate, updates[0].GetObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}

		creates := []core.CreateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() != "create" {
				continue
			}
			creates = append(creates, action.(core.CreateAction))
		}
		if test.expectCreate != nil {
			if len(creates) != 1 {
				t.Errorf("case %q: unexpected creates: %v", test.testName, creates)
			} else if e, a := test.expectCreate, creates[0].GetObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectCreate == nil && len(creates) > 0 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
		}

	}

}

func TestEmptySubsets(t *testing.T) {
	ns := metav1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}
	endpoints := &corev1.EndpointsList{
		Items: []corev1.Endpoints{{
			ObjectMeta: om("foo"),
			Subsets:    nil,
		}},
	}
	fakeClient := fake.NewSimpleClientset()
	if endpoints != nil {
		fakeClient = fake.NewSimpleClientset(endpoints)
	}
	epAdapter := reconcilers.NewEndpointsAdapter(fakeClient.CoreV1(), nil)
	reconciler := reconcilers.NewMasterCountEndpointReconciler(1, epAdapter)
	endpointPorts := []corev1.EndpointPort{
		{Name: "foo", Port: 8080, Protocol: "TCP"},
	}
	err := reconciler.RemoveEndpoints("foo", net.ParseIP("1.2.3.4"), endpointPorts)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCreateOrUpdateMasterService(t *testing.T) {
	singleStack := corev1.IPFamilyPolicySingleStack
	ns := metav1.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}

	createTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		expectCreate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service does not exist",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			expectCreate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					IPFamilyPolicy:  &singleStack,
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
	}
	for _, test := range createTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset()
		master.ServiceClient = fakeClient.CoreV1()
		master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, net.ParseIP("1.2.3.4"), test.servicePorts, test.serviceType, false)
		creates := []core.CreateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "create" {
				creates = append(creates, action.(core.CreateAction))
			}
		}
		if test.expectCreate != nil {
			if len(creates) != 1 {
				t.Errorf("case %q: unexpected creations: %v", test.testName, creates)
			} else {
				obj := creates[0].GetObject()
				if e, a := test.expectCreate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectCreate == nil && len(creates) > 1 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
		}
	}

	reconcileTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		service      *corev1.Service
		expectUpdate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8000, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition missing port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
				{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
						{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "bar", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port name",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect target port",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect protocol",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "UDP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition has incorrect type",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeNodePort,
				},
			},
			expectUpdate: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition satisfies",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range reconcileTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.ServiceClient = fakeClient.CoreV1()
		err := master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, net.ParseIP("1.2.3.4"), test.servicePorts, test.serviceType, true)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "update" {
				updates = append(updates, action.(core.UpdateAction))
			}
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else {
				obj := updates[0].GetObject()
				if e, a := test.expectUpdate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}

	nonReconcileTests := []struct {
		testName     string
		serviceName  string
		servicePorts []corev1.ServicePort
		serviceType  corev1.ServiceType
		service      *corev1.Service
		expectUpdate *corev1.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port, no expected update",
			serviceName: "foo",
			servicePorts: []corev1.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: corev1.ServiceTypeClusterIP,
			service: &corev1.Service{
				ObjectMeta: om("foo"),
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: corev1.ServiceAffinityNone,
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range nonReconcileTests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.ServiceClient = fakeClient.CoreV1()
		err := master.CreateOrUpdateMasterServiceIfNeeded(test.serviceName, net.ParseIP("1.2.3.4"), test.servicePorts, test.serviceType, false)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		updates := []core.UpdateAction{}
		for _, action := range fakeClient.Actions() {
			if action.GetVerb() == "update" {
				updates = append(updates, action.(core.UpdateAction))
			}
		}
		if test.expectUpdate != nil {
			if len(updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, updates)
			} else {
				obj := updates[0].GetObject()
				if e, a := test.expectUpdate.Spec, obj.(*corev1.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}
}
