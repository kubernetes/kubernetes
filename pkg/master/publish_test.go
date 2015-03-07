/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestEnsureEndpointsContain(t *testing.T) {
	tests := []struct {
		serviceName       string
		ip                string
		port              int
		expectError       bool
		expectUpdate      bool
		endpoints         *api.EndpointsList
		expectedEndpoints []api.Endpoint
		err               error
		masterCount       int
	}{
		{
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			expectError:  false,
			expectUpdate: true,
			masterCount:  1,
		},
		{
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			expectError:  false,
			expectUpdate: false,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
						},
						Endpoints: []api.Endpoint{
							{
								IP:   "1.2.3.4",
								Port: 8080,
							},
						},
						Protocol: api.ProtocolTCP,
					},
				},
			},
			masterCount:       1,
			expectedEndpoints: []api.Endpoint{{"1.2.3.4", 8080}},
		},
		{
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			expectError:  false,
			expectUpdate: true,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{
							Name:      "foo",
							Namespace: api.NamespaceDefault,
						},
						Endpoints: []api.Endpoint{
							{
								IP:   "4.3.2.1",
								Port: 8080,
							},
						},
						Protocol: api.ProtocolTCP,
					},
				},
			},
			masterCount: 1,
		},
		{
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			expectError:  false,
			expectUpdate: true,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{
							Name:      "foo",
							Namespace: api.NamespaceDefault,
						},
						Endpoints: []api.Endpoint{
							{
								IP:   "4.3.2.1",
								Port: 9090,
							},
						},
						Protocol: api.ProtocolTCP,
					},
				},
			},
			masterCount:       2,
			expectedEndpoints: []api.Endpoint{{"4.3.2.1", 9090}, {"1.2.3.4", 8080}},
		},
		{
			serviceName:  "foo",
			ip:           "1.2.3.4",
			port:         8080,
			expectError:  false,
			expectUpdate: true,
			endpoints: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{
							Name:      "foo",
							Namespace: api.NamespaceDefault,
						},
						Endpoints: []api.Endpoint{
							{
								IP:   "4.3.2.1",
								Port: 9090,
							},
							{
								IP:   "1.2.3.4",
								Port: 8000,
							},
						},
						Protocol: api.ProtocolTCP,
					},
				},
			},
			masterCount:       2,
			expectedEndpoints: []api.Endpoint{{"1.2.3.4", 8000}, {"1.2.3.4", 8080}},
		},
	}
	for _, test := range tests {
		master := Master{}
		registry := &registrytest.EndpointRegistry{
			Endpoints: test.endpoints,
			Err:       test.err,
		}
		master.endpointRegistry = registry
		master.masterCount = test.masterCount
		err := master.ensureEndpointsContain(test.serviceName, net.ParseIP(test.ip), test.port)
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectUpdate {
			if test.expectedEndpoints == nil {
				test.expectedEndpoints = []api.Endpoint{{test.ip, test.port}}
			}
			expectedUpdate := api.Endpoints{
				ObjectMeta: api.ObjectMeta{
					Name:      test.serviceName,
					Namespace: "default",
				},
				Endpoints: test.expectedEndpoints,
				Protocol:  "TCP",
			}
			if len(registry.Updates) != 1 {
				t.Errorf("unexpected updates: %v", registry.Updates)
			} else if !reflect.DeepEqual(expectedUpdate, registry.Updates[0]) {
				t.Errorf("expected update:\n%#v\ngot:\n%#v\n", expectedUpdate, registry.Updates[0])
			}
		}
		if !test.expectUpdate && len(registry.Updates) > 0 {
			t.Errorf("no update expected, yet saw: %v", registry.Updates)
		}
	}
}

func TestEnsureEndpointsContainConverges(t *testing.T) {
	master := Master{}
	registry := &registrytest.EndpointRegistry{
		Endpoints: &api.EndpointsList{
			Items: []api.Endpoints{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
					},
					Endpoints: []api.Endpoint{
						{
							IP:   "4.3.2.1",
							Port: 9000,
						},
						{
							IP:   "1.2.3.4",
							Port: 8000,
						},
					},
					Protocol: api.ProtocolTCP,
				},
			},
		},
	}
	master.endpointRegistry = registry
	master.masterCount = 2
	// This is purposefully racy, it shouldn't matter the order that these things arrive,
	// we should still converge on the right answer.
	wg := sync.WaitGroup{}
	wg.Add(2)
	go func() {
		for i := 0; i < 10; i++ {
			if err := master.ensureEndpointsContain("foo", net.ParseIP("4.3.2.1"), 9090); err != nil {
				t.Errorf("unexpected error: %v", err)
				t.Fail()
			}
		}
		wg.Done()
	}()
	go func() {
		for i := 0; i < 10; i++ {
			if err := master.ensureEndpointsContain("foo", net.ParseIP("1.2.3.4"), 8080); err != nil {
				t.Errorf("unexpected error: %v", err)
				t.Fail()
			}
		}
		wg.Done()
	}()
	wg.Wait()

	// We should see at least two updates.
	if len(registry.Updates) > 2 {
		t.Errorf("unexpected updates: %v", registry.Updates)
	}
	// Pick up the last update and validate.
	endpoints := registry.Updates[len(registry.Updates)-1]
	if len(endpoints.Endpoints) != 2 {
		t.Errorf("unexpected update: %v", endpoints)
	}
	for _, endpoint := range endpoints.Endpoints {
		if endpoint.IP == "4.3.2.1" && endpoint.Port != 9090 {
			t.Errorf("unexpected endpoint state: %v", endpoint)
		}
		if endpoint.IP == "1.2.3.4" && endpoint.Port != 8080 {
			t.Errorf("unexpected endpoint state: %v", endpoint)
		}
	}
}
