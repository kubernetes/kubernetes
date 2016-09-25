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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getIngressResourceName() string {
	return "ingresses"
}

func TestListIngress(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &extensions.IngressList{
				Items: []extensions.Ingress{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: extensions.IngressSpec{
							Rules: []extensions.IngressRule{},
						},
					},
				},
			},
		},
	}
	receivedIngressList, err := c.Setup(t).Extensions().Ingress(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedIngressList, err)
}

func TestGetIngress(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Extensions().Ingress(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedIngress, err)
}

func TestGetIngressWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedIngress, err := c.Setup(t).Extensions().Ingress(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedIngress, err)
}

func TestUpdateIngress(t *testing.T) {
	ns := api.NamespaceDefault
	requestIngress := &extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Extensions().Ingress(ns).Update(requestIngress)
	defer c.Close()
	c.Validate(t, receivedIngress, err)
}

func TestUpdateIngressStatus(t *testing.T) {
	ns := api.NamespaceDefault
	lbStatus := api.LoadBalancerStatus{
		Ingress: []api.LoadBalancerIngress{
			{IP: "127.0.0.1"},
		},
	}
	requestIngress := &extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Status: extensions.IngressStatus{
			LoadBalancer: lbStatus,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{},
				},
				Status: extensions.IngressStatus{
					LoadBalancer: lbStatus,
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Extensions().Ingress(ns).UpdateStatus(requestIngress)
	defer c.Close()
	c.Validate(t, receivedIngress, err)
}

func TestDeleteIngress(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().Ingress(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateIngress(t *testing.T) {
	ns := api.NamespaceDefault
	requestIngress := &extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getIngressResourceName(), ns, ""),
			Body:   requestIngress,
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Extensions().Ingress(ns).Create(requestIngress)
	defer c.Close()
	c.Validate(t, receivedIngress, err)
}
