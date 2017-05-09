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

package v1

import (
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getResourceQuotasResoureName() string {
	return "resourcequotas"
}

func TestResourceQuotaCreate(t *testing.T) {
	ns := v1.NamespaceDefault
	resourceQuota := &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU:                    resource.MustParse("100"),
				v1.ResourceMemory:                 resource.MustParse("10000"),
				v1.ResourcePods:                   resource.MustParse("10"),
				v1.ResourceServices:               resource.MustParse("10"),
				v1.ResourceReplicationControllers: resource.MustParse("10"),
				v1.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   resourceQuota,
		},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}

	response, err := c.Setup(t).ResourceQuotas(ns).Create(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaGet(t *testing.T) {
	ns := v1.NamespaceDefault
	resourceQuota := &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU:                    resource.MustParse("100"),
				v1.ResourceMemory:                 resource.MustParse("10000"),
				v1.ResourcePods:                   resource.MustParse("10"),
				v1.ResourceServices:               resource.MustParse("10"),
				v1.ResourceReplicationControllers: resource.MustParse("10"),
				v1.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}

	response, err := c.Setup(t).ResourceQuotas(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestResourceQuotaList(t *testing.T) {
	ns := v1.NamespaceDefault

	resourceQuotaList := &v1.ResourceQuotaList{
		Items: []v1.ResourceQuota{
			{
				ObjectMeta: v1.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: resourceQuotaList},
	}
	response, err := c.Setup(t).ResourceQuotas(ns).List(labels.Everything())
	c.Validate(t, response, err)
}

func TestResourceQuotaUpdate(t *testing.T) {
	ns := v1.NamespaceDefault
	resourceQuota := &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Name:            "abc",
			Namespace:       "foo",
			ResourceVersion: "1",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU:                    resource.MustParse("100"),
				v1.ResourceMemory:                 resource.MustParse("10000"),
				v1.ResourcePods:                   resource.MustParse("10"),
				v1.ResourceServices:               resource.MustParse("10"),
				v1.ResourceReplicationControllers: resource.MustParse("10"),
				v1.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}
	response, err := c.Setup(t).ResourceQuotas(ns).Update(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaStatusUpdate(t *testing.T) {
	ns := v1.NamespaceDefault
	resourceQuota := &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Name:            "abc",
			Namespace:       "foo",
			ResourceVersion: "1",
		},
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU:                    resource.MustParse("100"),
				v1.ResourceMemory:                 resource.MustParse("10000"),
				v1.ResourcePods:                   resource.MustParse("10"),
				v1.ResourceServices:               resource.MustParse("10"),
				v1.ResourceReplicationControllers: resource.MustParse("10"),
				v1.ResourceQuotas:                 resource.MustParse("10"),
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, "abc") + "/status",
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: resourceQuota},
	}
	response, err := c.Setup(t).ResourceQuotas(ns).UpdateStatus(resourceQuota)
	c.Validate(t, response, err)
}

func TestResourceQuotaDelete(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath(getResourceQuotasResoureName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).ResourceQuotas(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestResourceQuotaWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getResourceQuotasResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).ResourceQuotas(v1.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
