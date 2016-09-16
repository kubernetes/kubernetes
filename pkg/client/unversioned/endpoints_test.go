/*
Copyright 2015 The Kubernetes Authors.

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

package unversioned_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func TestListEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Default.ResourcePath("endpoints", ns, ""), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200,
			Body: &api.EndpointsList{
				Items: []api.Endpoints{
					{
						ObjectMeta: api.ObjectMeta{Name: "endpoint-1"},
						Subsets: []api.EndpointSubset{{
							Addresses: []api.EndpointAddress{{IP: "10.245.1.2"}, {IP: "10.245.1.3"}},
							Ports:     []api.EndpointPort{{Port: 8080}},
						}},
					},
				},
			},
		},
	}
	receivedEndpointsList, err := c.Setup(t).Endpoints(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedEndpointsList, err)
}

func TestGetEndpoints(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "GET", Path: testapi.Default.ResourcePath("endpoints", ns, "endpoint-1"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "endpoint-1"}}},
	}
	response, err := c.Setup(t).Endpoints(ns).Get("endpoint-1")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestGetEndpointWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Endpoints(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}
