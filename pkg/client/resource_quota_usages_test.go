/*
Copyright 2014 Google Inc. All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func TestResourceQuotaUsageCreate(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuotaUsage := &api.ResourceQuotaUsage{
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
			Method: "POST",
			Path:   buildResourcePath(ns, "/resourceQuotaUsages"),
			Query:  buildQueryValues(ns, nil),
			Body:   resourceQuotaUsage,
		},
		Response: Response{StatusCode: 200, Body: resourceQuotaUsage},
	}

	err := c.Setup().ResourceQuotaUsages(ns).Create(resourceQuotaUsage)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestInvalidResourceQuotaUsageCreate(t *testing.T) {
	ns := api.NamespaceDefault
	resourceQuotaUsage := &api.ResourceQuotaUsage{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
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
			Method: "POST",
			Path:   buildResourcePath(ns, "/resourceQuotaUsages"),
			Query:  buildQueryValues(ns, nil),
			Body:   resourceQuotaUsage,
		},
		Response: Response{StatusCode: 200, Body: resourceQuotaUsage},
	}

	err := c.Setup().ResourceQuotaUsages(ns).Create(resourceQuotaUsage)
	if err == nil {
		t.Errorf("Expected error due to missing ResourceVersion")
	}
}
