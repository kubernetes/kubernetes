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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
	"k8s.io/kubernetes/pkg/labels"
)

func TestListServices(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Query:  simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200,
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
	receivedServiceList, err := c.Setup(t).Services(ns).List(api.ListOptions{})
	defer c.Close()
	t.Logf("received services: %v %#v", err, receivedServiceList)
	c.Validate(t, receivedServiceList, err)
}

func TestListServicesLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Query:  simple.BuildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: simple.Response{StatusCode: 200,
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
	defer c.Close()
	c.QueryValidator[labelSelectorQueryParamName] = simple.ValidateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	options := api.ListOptions{LabelSelector: selector}
	receivedServiceList, err := c.Services(ns).List(options)
	c.Validate(t, receivedServiceList, err)
}

func TestGetService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, "1"),
			Query:  simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup(t).Services(ns).Get("1")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestGetServiceWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Services(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestCreateService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath("services", ns, ""),
			Body:   &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}},
			Query:  simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}}},
	}
	response, err := c.Setup(t).Services(ns).Create(&api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1"}})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestUpdateService(t *testing.T) {
	ns := api.NamespaceDefault
	svc := &api.Service{ObjectMeta: api.ObjectMeta{Name: "service-1", ResourceVersion: "1"}}
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath("services", ns, "service-1"), Body: svc, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: svc},
	}
	response, err := c.Setup(t).Services(ns).Update(svc)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestDeleteService(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath("services", ns, "1"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Services(ns).Delete("1")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestUpdateServiceStatus(t *testing.T) {
	ns := api.NamespaceDefault
	lbStatus := api.LoadBalancerStatus{
		Ingress: []api.LoadBalancerIngress{
			{IP: "127.0.0.1"},
		},
	}
	requestService := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Status: api.ServiceStatus{
			LoadBalancer: lbStatus,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath("services", ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ServiceSpec{},
				Status: api.ServiceStatus{
					LoadBalancer: lbStatus,
				},
			},
		},
	}
	receivedService, err := c.Setup(t).Services(ns).UpdateStatus(requestService)
	defer c.Close()
	c.Validate(t, receivedService, err)
}

func TestServiceProxyGet(t *testing.T) {
	body := "OK"
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, "service-1") + "/proxy/foo",
			Query:  simple.BuildQueryValues(url.Values{"param-name": []string{"param-value"}}),
		},
		Response: simple.Response{StatusCode: 200, RawBody: &body},
	}
	response, err := c.Setup(t).Services(ns).ProxyGet("", "service-1", "", "foo", map[string]string{"param-name": "param-value"}).DoRaw()
	defer c.Close()
	c.ValidateRaw(t, response, err)

	// With scheme and port specified
	c = &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("services", ns, "https:service-1:my-port") + "/proxy/foo",
			Query:  simple.BuildQueryValues(url.Values{"param-name": []string{"param-value"}}),
		},
		Response: simple.Response{StatusCode: 200, RawBody: &body},
	}
	response, err = c.Setup(t).Services(ns).ProxyGet("https", "service-1", "my-port", "foo", map[string]string{"param-name": "param-value"}).DoRaw()
	defer c.Close()
	c.ValidateRaw(t, response, err)
}
