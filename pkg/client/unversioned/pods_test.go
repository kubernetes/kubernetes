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
	"net/http"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
	"k8s.io/kubernetes/pkg/labels"
)

func TestListEmptyPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: http.StatusOK, Body: &api.PodList{}},
	}
	podList, err := c.Setup(t).Pods(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, podList, err)
}

func TestListPods(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: http.StatusOK,
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
	receivedPodList, err := c.Setup(t).Pods(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedPodList, err)
}

func TestListPodsLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("pods", ns, ""),
			Query:  simple.BuildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: simple.Response{
			StatusCode: http.StatusOK,
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
	defer c.Close()
	c.QueryValidator[labelSelectorQueryParamName] = simple.ValidateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	options := api.ListOptions{LabelSelector: selector}
	receivedPodList, err := c.Pods(ns).List(options)
	c.Validate(t, receivedPodList, err)
}

func TestGetPod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: http.StatusOK,
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
	defer c.Close()
	c.Validate(t, receivedPod, err)
}

func TestGetPodWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Pods(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestDeletePod(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: http.StatusOK},
	}
	err := c.Setup(t).Pods(ns).Delete("foo", nil)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Default.ResourcePath("pods", ns, ""), Query: simple.BuildQueryValues(nil), Body: requestPod},
		Response: simple.Response{
			StatusCode: http.StatusOK,
			Body:       requestPod,
		},
	}
	receivedPod, err := c.Setup(t).Pods(ns).Create(requestPod)
	defer c.Close()
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
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath("pods", ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: http.StatusOK, Body: requestPod},
	}
	receivedPod, err := c.Setup(t).Pods(ns).Update(requestPod)
	defer c.Close()
	c.Validate(t, receivedPod, err)
}

func TestPodGetLogs(t *testing.T) {
	ns := api.NamespaceDefault
	opts := &api.PodLogOptions{
		Follow:     true,
		Timestamps: true,
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath("pods", ns, "podName") + "/log",
			Query: url.Values{
				"follow":     []string{"true"},
				"timestamps": []string{"true"},
			},
		},
		Response: simple.Response{StatusCode: http.StatusOK},
	}

	body, err := c.Setup(t).Pods(ns).GetLogs("podName", opts).Stream()
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer body.Close()
	c.ValidateCommon(t, err)
}
