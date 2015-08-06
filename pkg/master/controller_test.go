/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func TestSetEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	om := func(name string) api.ObjectMeta {
		return api.ObjectMeta{Namespace: ns, Name: name}
	}
	tests := []struct {
		testName          string
		serviceName       string
		ip                string
		port              int
		additionalMasters int
		endpoints         *api.EndpointsList
		expectUpdate      *api.Endpoints // nil means none expected
	}{
		{
			testName:    "no existing endpoints",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints:   nil,
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints satisfy",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
		},
		{
			testName:    "existing endpoints satisfy but too many",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters",
			serviceName:       "foo",
			ip:                "1.2.3.4",
			port:              8080,
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
						Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
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
					Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:          "existing endpoints satisfy but too many + extra masters + delete first",
			serviceName:       "foo",
			ip:                "4.3.2.4",
			port:              8080,
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
						Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
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
					Ports: []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints wrong name",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("bar"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints wrong IP",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints wrong port",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 9090, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
		{
			testName:    "existing endpoints wrong protocol",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: om("foo"),
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "UDP"}},
					}},
				}},
			},
			expectUpdate: &api.Endpoints{
				ObjectMeta: om("foo"),
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
					Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
				}},
			},
		},
	}
	for _, test := range tests {
		master := Controller{MasterCount: test.additionalMasters + 1}
		registry := &registrytest.EndpointRegistry{
			Endpoints: test.endpoints,
		}
		master.EndpointRegistry = registry
		err := master.SetEndpoints(test.serviceName, net.ParseIP(test.ip), test.port)
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
	}
}
