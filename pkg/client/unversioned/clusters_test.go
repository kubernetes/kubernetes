/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	. "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

import (
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
)

func getClustersResourceName() string {
	return "clusters"
}

func TestListClusterss(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", ""),
		},
		Response: simple.Response{StatusCode: 200, Body: &api.ClusterList{ListMeta: unversioned.ListMeta{ResourceVersion: "1"}}},
	}
	response, err := c.Setup(t).Clusters().List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestListClustersLabels(t *testing.T) {
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", ""),
			Query:  simple.BuildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.ClusterList{
				Items: []api.Cluster{
					{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
					},
				},
			},
		},
	}
	c.Setup(t)
	defer c.Close()
	c.QueryValidator[labelSelectorQueryParamName] = simple.ValidateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	options := api.ListOptions{LabelSelector: selector}
	receivedClusterList, err := c.Clusters().List(options)
	c.Validate(t, receivedClusterList, err)
}

func TestGetCluster(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", "1"),
		},
		Response: simple.Response{StatusCode: 200, Body: &api.Cluster{ObjectMeta: api.ObjectMeta{Name: "node-1"}}},
	}
	response, err := c.Setup(t).Clusters().Get("1")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestGetClusterWithNoName(t *testing.T) {
	c := &simple.Client{Error: true}
	receivedCluster, err := c.Setup(t).Clusters().Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedCluster, err)
}

func TestCreateCluster(t *testing.T) {
	requestCluster := &api.Cluster{
		ObjectMeta: api.ObjectMeta{
			Name: "cluster-1",
		},
		Status: api.ClusterStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1000m"),
				api.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
		Spec: api.ClusterSpec{},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", ""),
			Body:   requestCluster},
		Response: simple.Response{
			StatusCode: 200,
			Body:       requestCluster,
		},
	}
	receivedCluster, err := c.Setup(t).Clusters().Create(requestCluster)
	defer c.Close()
	c.Validate(t, receivedCluster, err)
}

func TestDeleteCluster(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", "foo"),
		},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Clusters().Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestUpdateCluster(t *testing.T) {
	requestCluster := &api.Cluster{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Status: api.ClusterStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1000m"),
				api.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
		Spec: api.ClusterSpec{},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getClustersResourceName(), "", "foo"),
		},
		Response: simple.Response{StatusCode: 200, Body: requestCluster},
	}
	response, err := c.Setup(t).Clusters().Update(requestCluster)
	defer c.Close()
	c.Validate(t, response, err)
}
