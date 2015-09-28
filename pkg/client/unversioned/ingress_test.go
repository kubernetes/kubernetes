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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getIngressResourceName() string {
	return "ingress"
}

func TestListIngress(t *testing.T) {
	ns := api.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, ""),
		},
		Response: Response{StatusCode: 200,
			Body: &experimental.IngressList{
				Items: []experimental.Ingress{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: experimental.IngressSpec{
							Rules: []experimental.IngressRule{},
						},
					},
				},
			},
		},
	}
	receivedIngressList, err := c.Setup(t).Experimental().Ingress(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, receivedIngressList, err)
}

func TestGetIngress(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  buildQueryValues(nil),
		},
		Response: Response{
			StatusCode: 200,
			Body: &experimental.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: experimental.IngressSpec{
					Rules: []experimental.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Experimental().Ingress(ns).Get("foo")
	c.Validate(t, receivedIngress, err)
}

func TestGetIngressWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedIngress, err := c.Setup(t).Experimental().Ingress(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedIngress, err)
}

func TestUpdateIngress(t *testing.T) {
	ns := api.NamespaceDefault
	requestIngress := &experimental.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  buildQueryValues(nil),
		},
		Response: Response{
			StatusCode: 200,
			Body: &experimental.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: experimental.IngressSpec{
					Rules: []experimental.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Experimental().Ingress(ns).Update(requestIngress)
	c.Validate(t, receivedIngress, err)
}

func TestUpdateIngressStatus(t *testing.T) {
	ns := api.NamespaceDefault
	lbStatus := api.LoadBalancerStatus{
		Ingress: []api.LoadBalancerIngress{
			{IP: "127.0.0.1"},
		},
	}
	requestIngress := &experimental.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Status: experimental.IngressStatus{
			LoadBalancer: lbStatus,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, "foo") + "/status",
			Query:  buildQueryValues(nil),
		},
		Response: Response{
			StatusCode: 200,
			Body: &experimental.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: experimental.IngressSpec{
					Rules: []experimental.IngressRule{},
				},
				Status: experimental.IngressStatus{
					LoadBalancer: lbStatus,
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Experimental().Ingress(ns).UpdateStatus(requestIngress)
	c.Validate(t, receivedIngress, err)
}

func TestDeleteIngress(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "DELETE",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, "foo"),
			Query:  buildQueryValues(nil),
		},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Experimental().Ingress(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestCreateIngress(t *testing.T) {
	ns := api.NamespaceDefault
	requestIngress := &experimental.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Experimental.ResourcePath(getIngressResourceName(), ns, ""),
			Body:   requestIngress,
			Query:  buildQueryValues(nil),
		},
		Response: Response{
			StatusCode: 200,
			Body: &experimental.Ingress{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: experimental.IngressSpec{
					Rules: []experimental.IngressRule{},
				},
			},
		},
	}
	receivedIngress, err := c.Setup(t).Experimental().Ingress(ns).Create(requestIngress)
	c.Validate(t, receivedIngress, err)
}
