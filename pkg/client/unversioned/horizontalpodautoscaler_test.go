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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getHorizontalPodAutoscalersResoureName() string {
	return "horizontalpodautoscalers"
}

func TestHorizontalPodAutoscalerCreate(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   &horizontalPodAutoscaler,
		},
		Response:      simple.Response{StatusCode: 200, Body: &horizontalPodAutoscaler},
		ResourceGroup: autoscaling.GroupName,
	}

	response, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).Create(&horizontalPodAutoscaler)
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerGet(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: autoscaling.GroupName,
	}

	response, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).Get("abc")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerList(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscalerList := &autoscaling.HorizontalPodAutoscalerList{
		Items: []autoscaling.HorizontalPodAutoscaler{
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
			Path:   testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscalerList},
		ResourceGroup: autoscaling.GroupName,
	}
	response, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:       simple.Request{Method: "PUT", Path: testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: autoscaling.GroupName,
	}
	response, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).Update(horizontalPodAutoscaler)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdateStatus(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request:       simple.Request{Method: "PUT", Path: testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc") + "/status", Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200, Body: horizontalPodAutoscaler},
		ResourceGroup: autoscaling.GroupName,
	}
	response, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).UpdateStatus(horizontalPodAutoscaler)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:       simple.Request{Method: "DELETE", Path: testapi.Autoscaling.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: autoscaling.GroupName,
	}
	err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestHorizontalPodAutoscalerWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Autoscaling.ResourcePathWithPrefix("watch", getHorizontalPodAutoscalersResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: autoscaling.GroupName,
	}
	_, err := c.Setup(t).Autoscaling().HorizontalPodAutoscalers(api.NamespaceAll).Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}
