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

package master

import (
	"net"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestReconcileEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	om := func(name string) api.ObjectMeta {
		return api.ObjectMeta{Namespace: ns, Name: name}
	}
	reconcile_tests := []struct {
		testName          string
		serviceName       string
		ip                string
		endpointPorts     []api.EndpointPort
		additionalMasters int
		endpoints         *api.EndpointsList
		expectUpdate      *api.Endpoints // nil means none expected
		expectCreate      *api.Endpoints // nil means none expected
	}{
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectCreate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
		{
			testName:      "existing endpoints satisfy but too many",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters",
			serviceName:       "foo",
			ip:                "1.2.3.4",
			endpointPorts:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{
						{IP: "1.2.3.4"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters + delete first",
			serviceName:       "foo",
			ip:                "4.3.2.4",
			endpointPorts:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{
							{IP: "1.2.3.4"},
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
							{IP: "4.3.2.3"},
							{IP: "4.3.2.4"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
						{IP: "4.3.2.3"},
						{IP: "4.3.2.4"},
					},
					Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy and endpoint addresses length less than master count",
			serviceName:       "foo",
			ip:                "4.3.2.2",
			endpointPorts:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{
							{IP: "4.3.2.1"},
							{IP: "4.3.2.2"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: nil,
		},
		{
			testName:          "existing endpoints current IP missing and address length less than master count",
			serviceName:       "foo",
			ip:                "4.3.2.2",
			endpointPorts:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			additionalMasters: 3,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{
							{IP: "4.3.2.1"},
						},
						Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{
						{IP: "4.3.2.1"},
						{IP: "4.3.2.2"},
					},
					Ports: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("bar"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectCreate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong IP",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 9090, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong protocol",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "UDP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "existing endpoints wrong port name",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "baz", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints extra service ports satisfy",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []api.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
				{Name: "baz", Port: 1010, Protocol: "TCP"},
			},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports: []api.EndpointPort{
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
			endpointPorts: []api.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports: []api.EndpointPort{
						{Name: "foo", Port: 8080, Protocol: "TCP"},
						{Name: "bar", Port: 1000, Protocol: "TCP"},
					},
				}},
			},
		},
	}
	for _, test := range reconcile_tests {
		fakeClient := fake.NewSimpleClientset()
		if test.endpoints != nil {
			fakeClient = fake.NewSimpleClientset(test.endpoints)
		}
		reconciler := NewMasterCountEndpointReconciler(test.additionalMasters+1, fakeClient.Core())
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

	non_reconcile_tests := []struct {
		testName          string
		serviceName       string
		ip                string
		endpointPorts     []api.EndpointPort
		additionalMasters int
		endpoints         *api.EndpointsList
		expectUpdate      *api.Endpoints // nil means none expected
		expectCreate      *api.Endpoints // nil means none expected
	}{
		{
			testName:    "existing endpoints extra service ports missing port no update",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []api.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: nil,
		},
		{
			testName:    "existing endpoints extra service ports, wrong ports, wrong IP",
			serviceName: "foo",
			ip:          "1.2.3.4",
			endpointPorts: []api.EndpointPort{
				{Name: "foo", Port: 8080, Protocol: "TCP"},
				{Name: "bar", Port: 1000, Protocol: "TCP"},
			},
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectCreate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
	}
	for _, test := range non_reconcile_tests {
		fakeClient := fake.NewSimpleClientset()
		if test.endpoints != nil {
			fakeClient = fake.NewSimpleClientset(test.endpoints)
		}
		reconciler := NewMasterCountEndpointReconciler(test.additionalMasters+1, fakeClient.Core())
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

func TestCreateOrUpdateMasterService(t *testing.T) {
	ns := api.NamespaceDefault
	om := func(name string) api.ObjectMeta {
		return api.ObjectMeta{Namespace: ns, Name: name}
	}

	create_tests := []struct {
		testName     string
		serviceName  string
		servicePorts []api.ServicePort
		serviceType  api.ServiceType
		expectCreate *api.Service // nil means none expected
	}{
		{
			testName:    "service does not exist",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			expectCreate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
	}
	for _, test := range create_tests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset()
		master.ServiceClient = fakeClient.Core()
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
				if e, a := test.expectCreate.Spec, obj.(*api.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected create:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectCreate == nil && len(creates) > 1 {
			t.Errorf("case %q: no create expected, yet saw: %v", test.testName, creates)
		}
	}

	reconcile_tests := []struct {
		testName     string
		serviceName  string
		servicePorts []api.ServicePort
		serviceType  api.ServiceType
		service      *api.Service
		expectUpdate *api.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8000, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition missing port",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
				{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
						{Name: "baz", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "bar", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect port name",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "UDP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect target port",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition incorrect protocol",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "UDP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition has incorrect type",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeNodePort,
				},
			},
			expectUpdate: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
		{
			testName:    "service definition satisfies",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range reconcile_tests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.ServiceClient = fakeClient.Core()
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
				if e, a := test.expectUpdate.Spec, obj.(*api.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}

	non_reconcile_tests := []struct {
		testName     string
		serviceName  string
		servicePorts []api.ServicePort
		serviceType  api.ServiceType
		service      *api.Service
		expectUpdate *api.Service // nil means none expected
	}{
		{
			testName:    "service definition wrong port, no expected update",
			serviceName: "foo",
			servicePorts: []api.ServicePort{
				{Name: "foo", Port: 8080, Protocol: "TCP", TargetPort: intstr.FromInt(8080)},
			},
			serviceType: api.ServiceTypeClusterIP,
			service: &api.Service{
				ObjectMeta: om("foo"),
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{Name: "foo", Port: 1000, Protocol: "TCP", TargetPort: intstr.FromInt(1000)},
					},
					Selector:        nil,
					ClusterIP:       "1.2.3.4",
					SessionAffinity: api.ServiceAffinityClientIP,
					Type:            api.ServiceTypeClusterIP,
				},
			},
			expectUpdate: nil,
		},
	}
	for _, test := range non_reconcile_tests {
		master := Controller{}
		fakeClient := fake.NewSimpleClientset(test.service)
		master.ServiceClient = fakeClient.Core()
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
				if e, a := test.expectUpdate.Spec, obj.(*api.Service).Spec; !reflect.DeepEqual(e, a) {
					t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
				}
			}
		}
		if test.expectUpdate == nil && len(updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, updates)
		}
	}
}
