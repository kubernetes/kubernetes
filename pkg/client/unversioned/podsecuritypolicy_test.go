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
	"fmt"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func TestPodSecurityPolicyCreate(t *testing.T) {
	ns := api.NamespaceNone
	psp := &extensions.PodSecurityPolicy{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
	}

	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getPSPResourcename(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   psp,
		},
		Response: simple.Response{StatusCode: 200, Body: psp},
	}

	response, err := c.Setup(t).PodSecurityPolicies().Create(psp)
	c.Validate(t, response, err)
}

func TestPodSecurityPolicyGet(t *testing.T) {
	ns := api.NamespaceNone
	psp := &extensions.PodSecurityPolicy{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getPSPResourcename(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: psp},
	}

	response, err := c.Setup(t).PodSecurityPolicies().Get("abc")
	c.Validate(t, response, err)
}

func TestPodSecurityPolicyList(t *testing.T) {
	ns := api.NamespaceNone
	pspList := &extensions.PodSecurityPolicyList{
		Items: []extensions.PodSecurityPolicy{
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
			Path:   testapi.Extensions.ResourcePath(getPSPResourcename(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: pspList},
	}
	response, err := c.Setup(t).PodSecurityPolicies().List(api.ListOptions{})
	c.Validate(t, response, err)
}

func TestPodSecurityPolicyUpdate(t *testing.T) {
	ns := api.NamespaceNone
	psp := &extensions.PodSecurityPolicy{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Extensions.ResourcePath(getPSPResourcename(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: psp},
	}
	response, err := c.Setup(t).PodSecurityPolicies().Update(psp)
	c.Validate(t, response, err)
}

func TestPodSecurityPolicyDelete(t *testing.T) {
	ns := api.NamespaceNone
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Extensions.ResourcePath(getPSPResourcename(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).PodSecurityPolicies().Delete("foo")
	c.Validate(t, nil, err)
}

func TestPodSecurityPolicyWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   fmt.Sprintf("%s/watch/%s", testapi.Extensions.ResourcePath("", "", ""), getPSPResourcename()),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PodSecurityPolicies().Watch(api.ListOptions{})
	c.Validate(t, nil, err)
}

func getPSPResourcename() string {
	return "podsecuritypolicies"
}
