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

package unversioned

import (
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func TestNamespaceCreate(t *testing.T) {
	// we create a namespace relative to another namespace
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath("namespaces", "", ""),
			Body:   namespace,
		},
		Response: Response{StatusCode: 200, Body: namespace},
	}

	// from the source ns, provision a new global namespace "foo"
	response, err := c.Setup(t).Namespaces().Create(namespace)

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if e, a := response.Name, namespace.Name; e != a {
		t.Errorf("%#v != %#v.", e, a)
	}
}

func TestNamespaceGet(t *testing.T) {
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo"),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: namespace},
	}

	response, err := c.Setup(t).Namespaces().Get("foo")

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if e, r := response.Name, namespace.Name; e != r {
		t.Errorf("%#v != %#v.", e, r)
	}
}

func TestNamespaceList(t *testing.T) {
	namespaceList := &api.NamespaceList{
		Items: []api.Namespace{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("namespaces", "", ""),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: namespaceList},
	}
	response, err := c.Setup(t).Namespaces().List(labels.Everything(), fields.Everything())

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if len(response.Items) != 1 {
		t.Errorf("%#v response.Items should have len 1.", response.Items)
	}

	responseNamespace := response.Items[0]
	if e, r := responseNamespace.Name, "foo"; e != r {
		t.Errorf("%#v != %#v.", e, r)
	}
}

func TestNamespaceUpdate(t *testing.T) {
	requestNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo")},
		Response: Response{StatusCode: 200, Body: requestNamespace},
	}
	receivedNamespace, err := c.Setup(t).Namespaces().Update(requestNamespace)
	c.Validate(t, receivedNamespace, err)
}

func TestNamespaceFinalize(t *testing.T) {
	requestNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{api.FinalizerKubernetes},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo") + "/finalize",
		},
		Response: Response{StatusCode: 200, Body: requestNamespace},
	}
	receivedNamespace, err := c.Setup(t).Namespaces().Finalize(requestNamespace)
	c.Validate(t, receivedNamespace, err)
}

func TestNamespaceDelete(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath("namespaces", "", "foo")},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Namespaces().Delete("foo")
	c.Validate(t, nil, err)
}

func TestNamespaceWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", "namespaces", "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Namespaces().Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
