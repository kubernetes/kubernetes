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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestSetEndpoints(t *testing.T) {
	tests := []struct {
		testName     string
		serviceName  string
		ip           string
		port         int
		endpoints    *api.EndpointsList
		expectUpdate bool
	}{
		{
			testName:     "no existing endpoints",
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			endpoints:    nil,
			expectUpdate: true,
		},
		{
			testName:    "existing endpoints satisfy",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: false,
		},
		{
			testName:    "existing endpoints satisfy but too many",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}, {IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: true,
		},
		{
			testName:    "existing endpoints wrong name",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "bar"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: true,
		},
		{
			testName:    "existing endpoints wrong IP",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "4.3.2.1"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: true,
		},
		{
			testName:    "existing endpoints wrong port",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 9090, Protocol: "TCP"}},
					}},
				}},
			},
			expectUpdate: true,
		},
		{
			testName:    "existing endpoints wrong protocol",
			serviceName: "foo",
			ip:          "1.2.3.4",
			port:        8080,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Subsets: []api.EndpointSubset{{
						Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
						Ports:     []api.EndpointPort{{Port: 8080, Protocol: "UDP"}},
					}},
				}},
			},
			expectUpdate: true,
		},
	}
	for _, test := range tests {
		master := Master{}
		registry := &registrytest.EndpointRegistry{
			Endpoints: test.endpoints,
		}
		master.endpointRegistry = registry
		err := master.setEndpoints(test.serviceName, net.ParseIP(test.ip), test.port)
		if err != nil {
			t.Errorf("case %q: unexpected error: %v", test.testName, err)
		}
		if test.expectUpdate {
			expectedSubsets := []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []api.EndpointPort{{Port: 8080, Protocol: "TCP"}},
			}}
			if len(registry.Updates) != 1 {
				t.Errorf("case %q: unexpected updates: %v", test.testName, registry.Updates)
			} else if !reflect.DeepEqual(expectedSubsets, registry.Updates[0].Subsets) {
				t.Errorf("case %q: expected update:\n%#v\ngot:\n%#v\n", test.testName, expectedSubsets, registry.Updates[0].Subsets)
			}
		}
		if !test.expectUpdate && len(registry.Updates) > 0 {
			t.Errorf("case %q: no update expected, yet saw: %v", test.testName, registry.Updates)
		}
	}
}
