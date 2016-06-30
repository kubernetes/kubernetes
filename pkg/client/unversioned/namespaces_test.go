/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func TestNamespaceCreate(t *testing.T) {
	// we create a namespace relative to another namespace
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath("namespaces", "", ""),
			Body:   namespace,
		},
		Response: simple.Response{StatusCode: 200, Body: namespace},
	}

	// from the source ns, provision a new global namespace "foo"
	response, err := c.Setup(t).Namespaces().Create(namespace)
	defer c.Close()

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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo"),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: namespace},
	}

	response, err := c.Setup(t).Namespaces().Get("foo")
	defer c.Close()

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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("namespaces", "", ""),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: namespaceList},
	}
	response, err := c.Setup(t).Namespaces().List(api.ListOptions{})
	defer c.Close()

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
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo")},
		Response: simple.Response{StatusCode: 200, Body: requestNamespace},
	}
	receivedNamespace, err := c.Setup(t).Namespaces().Update(requestNamespace)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath("namespaces", "", "foo") + "/finalize",
		},
		Response: simple.Response{StatusCode: 200, Body: requestNamespace},
	}
	receivedNamespace, err := c.Setup(t).Namespaces().Finalize(requestNamespace)
	defer c.Close()
	c.Validate(t, receivedNamespace, err)
}

func TestNamespaceDelete(t *testing.T) {
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath("namespaces", "", "foo")},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Namespaces().Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestNamespaceWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", "namespaces", "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Namespaces().Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}
