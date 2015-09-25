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
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getHorizontalPodAutoscalersResoureName() string {
	return "horizontalpodautoscalers"
}

func TestHorizontalPodAutoscalerCreate(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   &horizontalPodAutoscaler,
		},
		Response: Response{StatusCode: 200, Body: &horizontalPodAutoscaler},
	}

	response, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).Create(&horizontalPodAutoscaler)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerGet(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: horizontalPodAutoscaler},
	}

	response, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerList(t *testing.T) {
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
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: horizontalPodAutoscalerList},
	}
	response, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: horizontalPodAutoscaler},
	}
	response, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).Update(horizontalPodAutoscaler)
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerUpdateStatus(t *testing.T) {
	ns := api.NamespaceDefault
	horizontalPodAutoscaler := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "abc") + "/status", Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: horizontalPodAutoscaler},
	}
	response, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).UpdateStatus(horizontalPodAutoscaler)
	c.Validate(t, response, err)
}

func TestHorizontalPodAutoscalerDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Extensions.ResourcePath(getHorizontalPodAutoscalersResoureName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().HorizontalPodAutoscalers(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestHorizontalPodAutoscalerWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePathWithPrefix("watch", getHorizontalPodAutoscalersResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Extensions().HorizontalPodAutoscalers(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
