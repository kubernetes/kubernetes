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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"

	"net/url"
)

func TestSecurityContextConstraintsCreate(t *testing.T) {
	ns := api.NamespaceNone
	scc := &api.SecurityContextConstraints{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
	}

	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getSCCResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   scc,
		},
		Response: simple.Response{StatusCode: 200, Body: scc},
	}

	response, err := c.Setup(t).SecurityContextConstraints().Create(scc)
	c.Validate(t, response, err)
}

func TestSecurityContextConstraintsGet(t *testing.T) {
	ns := api.NamespaceNone
	scc := &api.SecurityContextConstraints{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getSCCResoureName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: scc},
	}

	response, err := c.Setup(t).SecurityContextConstraints().Get("abc")
	c.Validate(t, response, err)
}

func TestSecurityContextConstraintsList(t *testing.T) {
	ns := api.NamespaceNone
	sccList := &api.SecurityContextConstraintsList{
		Items: []api.SecurityContextConstraints{
			{
				ObjectMeta: api.ObjectMeta{
					Name: "abc",
				},
			},
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getSCCResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: sccList},
	}
	response, err := c.Setup(t).SecurityContextConstraints().List(api.ListOptions{})
	c.Validate(t, response, err)
}

func TestSecurityContextConstraintsUpdate(t *testing.T) {
	ns := api.NamespaceNone
	scc := &api.SecurityContextConstraints{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getSCCResoureName(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: scc},
	}
	response, err := c.Setup(t).SecurityContextConstraints().Update(scc)
	c.Validate(t, response, err)
}

func TestSecurityContextConstraintsDelete(t *testing.T) {
	ns := api.NamespaceNone
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath(getSCCResoureName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).SecurityContextConstraints().Delete("foo")
	c.Validate(t, nil, err)
}

func TestSecurityContextConstraintsWatch(t *testing.T) {
	ns := api.NamespaceNone
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getSCCResoureName(), ns, ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).SecurityContextConstraints().Watch(api.ListOptions{})
	c.Validate(t, nil, err)
}

func getSCCResoureName() string {
	return "securitycontextconstraints"
}
