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

package registry

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

func TestListPodsError(t *testing.T) {
	mockRegistry := MockPodRegistry{
		err: fmt.Errorf("test error"),
	}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
	}
	pods, err := storage.List(labels.Everything())
	if err != mockRegistry.err {
		t.Errorf("Expected %#v, Got %#v", mockRegistry.err, err)
	}
	if len(pods.(api.PodList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", pods)
	}
}

func TestListEmptyPodList(t *testing.T) {
	mockRegistry := MockPodRegistry{}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
	}
	pods, err := storage.List(labels.Everything())
	expectNoError(t, err)
	if len(pods.(api.PodList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", pods)
	}
}

func TestListPodList(t *testing.T) {
	mockRegistry := MockPodRegistry{
		pods: []api.Pod{
			{
				JSONBase: api.JSONBase{
					ID: "foo",
				},
			},
			{
				JSONBase: api.JSONBase{
					ID: "bar",
				},
			},
		},
	}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
	}
	podsObj, err := storage.List(labels.Everything())
	pods := podsObj.(api.PodList)
	expectNoError(t, err)
	if len(pods.Items) != 2 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods.Items[0].ID != "foo" {
		t.Errorf("Unexpected pod: %#v", pods.Items[0])
	}
	if pods.Items[1].ID != "bar" {
		t.Errorf("Unexpected pod: %#v", pods.Items[1])
	}
}

func TestExtractJson(t *testing.T) {
	mockRegistry := MockPodRegistry{}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
	}
	pod := api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	}
	body, err := json.Marshal(pod)
	expectNoError(t, err)
	podOut, err := storage.Extract(string(body))
	expectNoError(t, err)
	// Extract adds in a kind
	pod.Kind = "cluster#pod"
	if !reflect.DeepEqual(pod, podOut) {
		t.Errorf("Expected %#v, found %#v", pod, podOut)
	}
}

func TestGetPod(t *testing.T) {
	mockRegistry := MockPodRegistry{
		pod: &api.Pod{
			JSONBase: api.JSONBase{ID: "foo"},
		},
	}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
	}
	obj, err := storage.Get("foo")
	pod := obj.(*api.Pod)
	expectNoError(t, err)
	if !reflect.DeepEqual(*mockRegistry.pod, *pod) {
		t.Errorf("Unexpected pod.  Expected %#v, Got %#v", *mockRegistry.pod, *pod)
	}
}

func TestGetPodCloud(t *testing.T) {
	fakeCloud := &cloudprovider.FakeCloud{}
	mockRegistry := MockPodRegistry{
		pod: &api.Pod{
			JSONBase: api.JSONBase{ID: "foo"},
		},
	}
	storage := PodRegistryStorage{
		registry: &mockRegistry,
		cloud:    fakeCloud,
	}
	obj, err := storage.Get("foo")
	pod := obj.(*api.Pod)
	expectNoError(t, err)
	if !reflect.DeepEqual(*mockRegistry.pod, *pod) {
		t.Errorf("Unexpected pod.  Expected %#v, Got %#v", *mockRegistry.pod, *pod)
	}
	if len(fakeCloud.Calls) != 1 || fakeCloud.Calls[0] != "ip-address" {
		t.Errorf("Unexpected calls: %#v", fakeCloud.Calls)
	}
}

func TestMakePodStatus(t *testing.T) {
	status := makePodStatus(map[string]interface{}{})
	if status != "Pending" {
		t.Errorf("Expected 'Pending', got '%s'", status)
	}

	status = makePodStatus(map[string]interface{}{
		"State": map[string]interface{}{
			"Running": false,
		},
	})

	if status != "Stopped" {
		t.Errorf("Expected 'Stopped', got '%s'", status)
	}

	status = makePodStatus(map[string]interface{}{
		"State": map[string]interface{}{
			"Running": true,
		},
	})

	if status != "Running" {
		t.Errorf("Expected 'Running', got '%s'", status)
	}
}
