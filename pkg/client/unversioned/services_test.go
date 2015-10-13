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

package unversioned

import (
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func TestListServices(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "name",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ServiceSpec{
							Selector: map[string]string{
								"one": "two",
							},
						},
					},
				},
			},
		},
	}
	receivedServiceList, err := c.Setup(t).Services(ns).List(labels.Everything(), fields.Everything())
	t.Logf("received services: %v %#v", err, receivedServiceList)
	c.Validate(t, receivedServiceList, err)
}

func TestListServicesLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testapi.Default.Version())
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Query:  buildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: Response{StatusCode: 200,
			Body: &api.ServiceList{
				Items: []api.Service{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "name",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ServiceSpec{
							Selector: map[string]string{
								"one": "two",
							},
						},
					},
				},
			},
		},
	}
	c.Setup(t)
	c.QueryValidator[labelSelectorQueryParamName] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedServiceList, err := c.Services(ns).List(selector, fields.Everything())
	c.Validate(t, receivedServiceList, err)
}

func TestGetService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, "1"),
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup(t).Services(ns).Get("1")
	c.Validate(t, response, err)
}

func TestGetServiceWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup(t).Services(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestCreateService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Body:   &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}},
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup(t).Services(ns).Create(&api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}})
	c.Validate(t, response, err)
}

func TestUpdateService(t *testing.T) {
	ns := api.NamespaceDefault
	svc := &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1", ResourceVersion: "1"}}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath("services", ns, "service-1"), Body: svc, Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: svc},
	}
	response, err := c.Setup(t).Services(ns).Update(svc)
	c.Validate(t, response, err)
}

func TestDeleteService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath("services", ns, "1"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Services(ns).Delete("1")
	c.Validate(t, nil, err)
}

func TestServiceProxyGet(t *testing.T) {
	body := "OK"
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("proxy", "services", ns, "service-1") + "/foo",
			Query:  buildQueryValues(url.Values{"param-name": []string{"param-value"}}),
		},
		Response: Response{StatusCode: 200, RawBody: &body},
	}
	response, err := c.Setup(t).Services(ns).ProxyGet("service-1", "foo", map[string]string{"param-name": "param-value"}).DoRaw()
	c.ValidateRaw(t, response, err)
}
