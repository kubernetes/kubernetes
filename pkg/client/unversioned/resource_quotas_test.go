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

package client

import (
	"net/url"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func getResourceQuotasResoureName() string {
	return "resourcequotas"
}

func TestResourceQuotaCreate(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("10000"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getResourceQuotasResoureName(), ns, ""),
			Query:  buildQueryValues(ns, nil),
			Body:   resourceQuota,
		},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}

	response, err := c.Setup().ResourceQuotas(ns).Create(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaGet(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("10000"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getResourceQuotasResoureName(), ns, "abc"),
			Query:  buildQueryValues(ns, nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}

	response, err := c.Setup().ResourceQuotas(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestResourceQuotaList(t *testing.T) {
	ns := api.NamespaceDefault

	resourceQuotaList := &api.ResourceQuotaList{
		Items: []api.ResourceQuota{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getResourceQuotasResoureName(), ns, ""),
			Query:  buildQueryValues(ns, nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: resourceQuotaList},
	}
	response, err := c.Setup().ResourceQuotas(ns).List(labels.Everything())
	c.Validate(t, response, err)
}

func TestResourceQuotaUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       "foo",
			ResourceVersion: "1",
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("10000"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getResourceQuotasResoureName(), ns, "abc"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}
	response, err := c.Setup().ResourceQuotas(ns).Update(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaStatusUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       "foo",
			ResourceVersion: "1",
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("10000"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.ResourcePath(getResourceQuotasResoureName(), ns, "abc") + "/status",
			Query:  buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}
	response, err := c.Setup().ResourceQuotas(ns).UpdateStatus(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getResourceQuotasResoureName(), ns, "foo"), Query: buildQueryValues(ns, nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().ResourceQuotas(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestResourceQuotaWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Version() + "/watch/" + getResourceQuotasResoureName(),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup().ResourceQuotas(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
