/*
Copyright 2017 The Kubernetes Authors.

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

/*
Original Source:
https://github.com/openshift/origin/blob/bb340c5dd5ff72718be86fb194dedc0faed7f4c7/pkg/cmd/server/election/lease_endpoint_reconciler_test.go
*/

import (
	"net"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

type fakeLeases struct {
	keys map[string]bool
}

var _ Leases = &fakeLeases{}

func newFakeLeases() *fakeLeases {
	return &fakeLeases{make(map[string]bool)}
}

func (f *fakeLeases) ListLeases() ([]string, error) {
	res := make([]string, 0, len(f.keys))
	for ip := range f.keys {
		res = append(res, ip)
	}
	return res, nil
}

func (f *fakeLeases) UpdateLease(ip string) error {
	f.keys[ip] = true
	return nil
}

func (f *fakeLeases) SetKeys(keys []string) {
	for _, ip := range keys {
		f.keys[ip] = false
	}
}

func (f *fakeLeases) GetUpdatedKeys() []string {
	res := []string{}
	for ip, updated := range f.keys {
		if updated {
			res = append(res, ip)
		}
	}
	return res
}

func TestLeaseEndpointReconciler(t *testing.T) {
	ns := api.NamespaceDefault
	om := func(name string) metav1.ObjectMeta {
		return metav1.ObjectMeta{Namespace: ns, Name: name}
	}
	reconcileTests := []struct {
		testName      string
		serviceName   string
		ip            string
		endpointPorts []api.EndpointPort
		endpointKeys  []string
		endpoints     *api.EndpointsList
		expectUpdate  *api.Endpoints // nil means none expected
	}{
		{
			testName:      "no existing endpoints",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpoints:     nil,
			expectUpdate: &api.Endpoints{
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
			testName:      "existing endpoints satisfy + refresh existing key",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4"},
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
			testName:      "existing endpoints satisfy but too many + extra masters",
			serviceName:   "foo",
			ip:            "1.2.3.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"1.2.3.4", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
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
			testName:      "existing endpoints satisfy but too many + extra masters + delete first",
			serviceName:   "foo",
			ip:            "4.3.2.4",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"4.3.2.1", "4.3.2.2", "4.3.2.3", "4.3.2.4"},
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
			testName:      "existing endpoints current IP missing",
			serviceName:   "foo",
			ip:            "4.3.2.2",
			endpointPorts: []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
			endpointKeys:  []string{"4.3.2.1"},
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
			expectUpdate: &api.Endpoints{
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
	for _, test := range reconcileTests {
		fakeLeases := newFakeLeases()
		fakeLeases.SetKeys(test.endpointKeys)
		registry := &registrytest.EndpointRegistry{
			Endpoints: test.endpoints,
		}
		r := NewLeaseEndpointReconciler(registry, fakeLeases)
		err := r.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, true)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		if test.expectUpdate != nil {
			if len(registry.Updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, registry.Updates)
			} else if e, a := test.expectUpdate, &registry.Updates[0]; !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectUpdate == nil && len(registry.Updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, registry.Updates)
		}
		if updatedKeys := fakeLeases.GetUpdatedKeys(); len(updatedKeys) != 1 || updatedKeys[0] != test.ip {
			t.Errorf("case %q: expected the master's IP to be refreshed, but the following IPs were refreshed instead: %v", test.testName, updatedKeys)
		}
	}

	nonReconcileTests := []struct {
		testName      string
		serviceName   string
		ip            string
		endpointPorts []api.EndpointPort
		endpointKeys  []string
		endpoints     *api.EndpointsList
		expectUpdate  *api.Endpoints // nil means none expected
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
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Name: "foo", Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
	}
	for _, test := range nonReconcileTests {
		fakeLeases := newFakeLeases()
		fakeLeases.SetKeys(test.endpointKeys)
		registry := &registrytest.EndpointRegistry{
			Endpoints: test.endpoints,
		}
		r := NewLeaseEndpointReconciler(registry, fakeLeases)
		err := r.ReconcileEndpoints(test.serviceName, net.ParseIP(test.ip), test.endpointPorts, false)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		if test.expectUpdate != nil {
			if len(registry.Updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, registry.Updates)
			} else if e, a := test.expectUpdate, &registry.Updates[0]; !reflect.DeepEqual(e, a) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, e, a)
			}
		}
		if test.expectUpdate == nil && len(registry.Updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, registry.Updates)
		}
		if updatedKeys := fakeLeases.GetUpdatedKeys(); len(updatedKeys) != 1 || updatedKeys[0] != test.ip {
			t.Errorf("case %q: expected the master's IP to be refreshed, but the following IPs were refreshed instead: %v", test.testName, updatedKeys)
		}
	}
}
