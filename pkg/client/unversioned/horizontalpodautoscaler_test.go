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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getHorizontalPodAutoscalersResoureName() string {
	return "horizontalpodautoscalers"
}

func getHPAClient(t *testing.T, c *simple.Client, ns, resourceGroup string) unversioned.HorizontalPodAutoscalerInterface {
	switch resourceGroup {
	case autoscaling.GroupName:
		return c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns)
	case extensions.GroupName:
		return c.Setup(t).Extensions().HorizontalPodAutoscalers(ns)
	default:
		t.Fatalf("Unknown group %v", resourceGroup)
	}
	return nil
}

func testHorizontalPodAutoscalerCreate(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   &horizontalPodAutoscaler,
		},
		Response:      simple.Response{StatusCode: 200, Body: &horizontalPodAutoscaler},
		ResourceGroup: resourceGroup,
	}

	response, err := getHPAClient(t, c, ns, resourceGroup).Create(&horizontalPodAutoscaler)
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerCreate(t *testing.T) {
	testHorizontalPodAutoscalerCreate(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerCreate(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerGet(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: resourceGroup,
	}

	response, err := getHPAClient(t, c, ns, resourceGroup).Get("abc")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerGet(t *testing.T) {
	testHorizontalPodAutoscalerGet(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerGet(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerList(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	horizontalPodAutoscalerList := &extensions.HorizontalPodAutoscalerList{
		Items: []extensions.HorizontalPodAutoscaler{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: ns,
				},
			},
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscalerList},
		ResourceGroup: resourceGroup,
	}
	response, err := getHPAClient(t, c, ns, resourceGroup).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerList(t *testing.T) {
	testHorizontalPodAutoscalerList(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerList(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerUpdate(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:       simple.Request{Method: "PUT", Path: group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: resourceGroup,
	}
	response, err := getHPAClient(t, c, ns, resourceGroup).Update(horizontalPodAutoscaler)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdate(t *testing.T) {
	testHorizontalPodAutoscalerUpdate(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerUpdate(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerUpdateStatus(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:       simple.Request{Method: "PUT", Path: group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc") + "/status", Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: resourceGroup,
	}
	response, err := getHPAClient(t, c, ns, resourceGroup).UpdateStatus(horizontalPodAutoscaler)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdateStatus(t *testing.T) {
	testHorizontalPodAutoscalerUpdateStatus(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerUpdateStatus(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerDelete(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:       simple.Request{Method: "DELETE", Path: group.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: resourceGroup,
	}
	err := getHPAClient(t, c, ns, resourceGroup).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestHorizontalPodAutoscalerDelete(t *testing.T) {
	testHorizontalPodAutoscalerDelete(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerDelete(t, testapi.Autoscaling, autoscaling.GroupName)
}

func testHorizontalPodAutoscalerWatch(t *testing.T, group testapi.TestGroup, resourceGroup string) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   group.ResourcePathWithPrefix("watch", getHorizontalPodAutoscalersResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: resourceGroup,
	}
	_, err := getHPAClient(t, c, api.NamespaceAll, resourceGroup).Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestHorizontalPodAutoscalerWatch(t *testing.T) {
	testHorizontalPodAutoscalerWatch(t, testapi.Extensions, extensions.GroupName)
	testHorizontalPodAutoscalerWatch(t, testapi.Autoscaling, autoscaling.GroupName)
}
