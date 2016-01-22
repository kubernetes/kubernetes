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

package unversioned_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	. "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

type Foo struct {
	Field string
}

func TestListGeneric(t *testing.T) {
	ns := api.NamespaceDefault
	gvk := unversioned.GroupVersionKind{Group: "company.com", Version: "v1", Kind: "Foo"}
	body := "foobar"
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   "/apis/company.com/v1/foos",
		},
		Response: simple.Response{
			StatusCode: 200,
			RawBody:    &body,
		},
	}
	data, err := c.Setup(t).Extensions().Generic(ns, gvk).List(api.ListOptions{})
	c.ValidateRaw(t, data, err)
}

func TestGetGeneric(t *testing.T) {
	ns := api.NamespaceDefault
	gvk := unversioned.GroupVersionKind{Group: "company.com", Version: "v1", Kind: "Foo"}
	body := "foobar"
	name := "baz"
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   "/apis/company.com/v1/foos/baz",
		},
		Response: simple.Response{
			StatusCode: 200,
			RawBody:    &body,
		},
	}
	data, err := c.Setup(t).Extensions().Generic(ns, gvk).Get(name)
	c.ValidateRaw(t, data, err)
}

func TestCreateGeneric(t *testing.T) {
	ns := api.NamespaceDefault
	gvk := unversioned.GroupVersionKind{Group: "company.com", Version: "v1", Kind: "Foo"}
	body := "foobar"
	name := "baz"
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   "/apis/company.com/v1/foos",
		},
		Response: simple.Response{
			StatusCode: 200,
			RawBody:    &body,
		},
	}
	data, err := c.Setup(t).Extensions().Generic(ns, gvk).Create(name, &Foo{Field: "test"})
	c.ValidateRaw(t, data, err)
}

func TestUpdateGeneric(t *testing.T) {
	ns := api.NamespaceDefault
	gvk := unversioned.GroupVersionKind{Group: "company.com", Version: "v1", Kind: "Foo"}
	body := "foobar"
	name := "baz"
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   "/apis/company.com/v1/foos/baz",
		},
		Response: simple.Response{
			StatusCode: 200,
			RawBody:    &body,
		},
	}
	data, err := c.Setup(t).Extensions().Generic(ns, gvk).Update(name, &Foo{Field: "test"})
	c.ValidateRaw(t, data, err)
}

func TestDeleteGeneric(t *testing.T) {
	ns := api.NamespaceDefault
	gvk := unversioned.GroupVersionKind{Group: "company.com", Version: "v1", Kind: "Foo"}
	body := "foobar"
	name := "baz"
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   "/apis/company.com/v1/foos/baz",
		},
		Response: simple.Response{
			StatusCode: 200,
			RawBody:    &body,
		},
	}
	data, err := c.Setup(t).Extensions().Generic(ns, gvk).Delete(name)
	c.ValidateRaw(t, data, err)
}
