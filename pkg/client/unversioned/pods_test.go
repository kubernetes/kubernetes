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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func TestListEmptyPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: &api.PodList{}},
	}
	podList, err := c.Setup(t).Pods(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, podList, err)
}

func TestListPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						Status: api.PodStatus{
							Phase: api.PodRunning,
						},
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
	receivedPodList, err := c.Setup(t).Pods(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, receivedPodList, err)
}

func TestListPodsLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := api.LabelSelectorQueryParam(testapi.Default.Version())
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("pods", ns, ""),
			Query:  buildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: Response{
			StatusCode: 200,
			Body: &api.PodList{
				Items: []api.Pod{
					{
						Status: api.PodStatus{
							Phase: api.PodRunning,
						},
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
	c.QueryValidator[labelSelectorQueryParamName] = validateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	receivedPodList, err := c.Pods(ns).List(selector, fields.Everything())
	c.Validate(t, receivedPodList, err)
}

func TestGetPod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &api.Pod{
				Status: api.PodStatus{
					Phase: api.PodRunning,
				},
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
			},
		},
	}
	receivedPod, err := c.Setup(t).Pods(ns).Get("foo")
	c.Validate(t, receivedPod, err)
}

func TestGetPodWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup(t).Pods(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestDeletePod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Pods(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestCreatePod(t *testing.T) {
	ns := api.NamespaceDefault
	requestPod := &api.Pod{
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: buildQueryValues(nil), Body: requestPod},
		Response: Response{
			StatusCode: 200,
			Body:       requestPod,
		},
	}
	receivedPod, err := c.Setup(t).Pods(ns).Create(requestPod)
	c.Validate(t, receivedPod, err)
}

func TestUpdatePod(t *testing.T) {
	ns := api.NamespaceDefault
	requestPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: requestPod},
	}
	receivedPod, err := c.Setup(t).Pods(ns).Update(requestPod)
	c.Validate(t, receivedPod, err)
}
