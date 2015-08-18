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

package client

import (
	"net/url"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	v1api "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func getLimitRangesResourceName() string {
	return "limitranges"
}

func TestLimitRangeCreate(t *testing.T) {
	ns := v1api.NamespaceDefault
	limitRange := &v1api.LimitRange{
		ObjectMeta: v1api.ObjectMeta{
			Name: "abc",
		},
		Spec: v1api.LimitRangeSpec{
			Limits: []v1api.LimitRangeItem{
				{
					Type: v1api.LimitTypePod,
					Max: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("100"),
						v1api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("0"),
						v1api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getLimitRangesResourceName(), ns, ""),
			Query:  buildQueryValues(ns, nil),
			Body:   limitRange,
		},
		Response: Response{StatusCode: 200, Body: limitRange},
	}

	response, err := c.Setup().LimitRanges(ns).Create(limitRange)
	c.Validate(t, response, err)
}

func TestLimitRangeGet(t *testing.T) {
	ns := v1api.NamespaceDefault
	limitRange := &v1api.LimitRange{
		ObjectMeta: v1api.ObjectMeta{
			Name: "abc",
		},
		Spec: v1api.LimitRangeSpec{
			Limits: []v1api.LimitRangeItem{
				{
					Type: v1api.LimitTypePod,
					Max: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("100"),
						v1api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("0"),
						v1api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getLimitRangesResourceName(), ns, "abc"),
			Query:  buildQueryValues(ns, nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: limitRange},
	}

	response, err := c.Setup().LimitRanges(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestLimitRangeList(t *testing.T) {
	ns := v1api.NamespaceDefault

	limitRangeList := &v1api.LimitRangeList{
		Items: []v1api.LimitRange{
			{
				ObjectMeta: v1api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getLimitRangesResourceName(), ns, ""),
			Query:  buildQueryValues(ns, nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: limitRangeList},
	}
	response, err := c.Setup().LimitRanges(ns).List(labels.Everything())
	c.Validate(t, response, err)
}

func TestLimitRangeUpdate(t *testing.T) {
	ns := v1api.NamespaceDefault
	limitRange := &v1api.LimitRange{
		ObjectMeta: v1api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: v1api.LimitRangeSpec{
			Limits: []v1api.LimitRangeItem{
				{
					Type: v1api.LimitTypePod,
					Max: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("100"),
						v1api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("0"),
						v1api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getLimitRangesResourceName(), ns, "abc"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: limitRange},
	}
	response, err := c.Setup().LimitRanges(ns).Update(limitRange)
	c.Validate(t, response, err)
}

func TestInvalidLimitRangeUpdate(t *testing.T) {
	ns := v1api.NamespaceDefault
	limitRange := &v1api.LimitRange{
		ObjectMeta: v1api.ObjectMeta{
			Name: "abc",
		},
		Spec: v1api.LimitRangeSpec{
			Limits: []v1api.LimitRangeItem{
				{
					Type: v1api.LimitTypePod,
					Max: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("100"),
						v1api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: v1api.ResourceList{
						v1api.ResourceCPU:    resource.MustParse("0"),
						v1api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getLimitRangesResourceName(), ns, "abc"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: limitRange},
	}
	_, err := c.Setup().LimitRanges(ns).Update(limitRange)
	if err == nil {
		t.Errorf("Expected an error due to missing ResourceVersion")
	}
}

func TestLimitRangeDelete(t *testing.T) {
	ns := v1api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getLimitRangesResourceName(), ns, "foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().LimitRanges(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestLimitRangeWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Version() + "/watch/" + getLimitRangesResourceName(),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup().LimitRanges(v1api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
