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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getLimitRangesResourceName() string {
	return "limitranges"
}

func TestLimitRangeCreate(t *testing.T) {
	ns := api.NamespaceDefault
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100"),
						api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("0"),
						api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   limitRange,
		},
		Response: Response{StatusCode: 200, Body: limitRange},
	}

	response, err := c.Setup(t).LimitRanges(ns).Create(limitRange)
	c.Validate(t, response, err)
}

func TestLimitRangeGet(t *testing.T) {
	ns := api.NamespaceDefault
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100"),
						api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("0"),
						api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: limitRange},
	}

	response, err := c.Setup(t).LimitRanges(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestLimitRangeList(t *testing.T) {
	ns := api.NamespaceDefault

	limitRangeList := &api.LimitRangeList{
		Items: []api.LimitRange{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: limitRangeList},
	}
	response, err := c.Setup(t).LimitRanges(ns).List(labels.Everything())
	c.Validate(t, response, err)
}

func TestLimitRangeUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100"),
						api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("0"),
						api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: limitRange},
	}
	response, err := c.Setup(t).LimitRanges(ns).Update(limitRange)
	c.Validate(t, response, err)
}

func TestInvalidLimitRangeUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100"),
						api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("0"),
						api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: limitRange},
	}
	_, err := c.Setup(t).LimitRanges(ns).Update(limitRange)
	if err == nil {
		t.Errorf("Expected an error due to missing ResourceVersion")
	}
}

func TestLimitRangeDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath(getLimitRangesResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).LimitRanges(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestLimitRangeWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getLimitRangesResourceName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).LimitRanges(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
