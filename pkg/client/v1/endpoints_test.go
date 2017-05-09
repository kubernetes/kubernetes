/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/labels"
)

func TestListEndpoints(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.Default.ResourcePath("endpoints", ns, ""), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200,
			Body: &v1.EndpointsList{
				Items: []v1.Endpoints{
					{
						ObjectMeta: v1.ObjectMeta{Name: "endpoint-1"},
						Subsets: []v1.EndpointSubset{{
							Addresses: []v1.EndpointAddress{{IP: "10.245.1.2"}, {IP: "10.245.1.3"}},
							Ports:     []v1.EndpointPort{{Port: 8080}},
						}},
					},
				},
			},
		},
	}
	receivedEndpointsList, err := c.Setup(t).Endpoints(ns).List(labels.Everything())
	c.Validate(t, receivedEndpointsList, err)
}

func TestGetEndpoints(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: testapi.Default.ResourcePath("endpoints", ns, "endpoint-1"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: &v1.Endpoints{ObjectMeta: v1.ObjectMeta{Name: "endpoint-1"}}},
	}
	response, err := c.Setup(t).Endpoints(ns).Get("endpoint-1")
	c.Validate(t, response, err)
}

func TestGetEndpointWithNoName(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup(t).Endpoints(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}
