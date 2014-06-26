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
	"fmt"
	"reflect"
	"strings"
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

func TestCreatePodIdError(t *testing.T) {
	storage := PodRegistryStorage{
		registry: &MockPodRegistry{},
	}
	pod := api.Pod{}
	_, err := storage.Create(pod)
	if !strings.HasPrefix(err.Error(), "id is unspecified: ") {
		t.Errorf("Expected id is unspecified error, Got %#v", err)
	}
}

type MockScheduler struct {
	err error
	pod api.Pod
}

func (m *MockScheduler) Schedule(pod api.Pod) (string, error) {
	m.pod = pod
	return "", m.err
}

func TestCreatePodContainerIdError(t *testing.T) {
	mockScheduler := MockScheduler{
		err: fmt.Errorf("test error"),
	}
	storage := PodRegistryStorage{
		scheduler: &mockScheduler,
	}
	pod := api.Pod{
		JSONBase: api.JSONBase{
			ID: "test",
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Containers: []api.Container{
					api.Container{},
				},
			},
		},
	}
	_, err := storage.Create(pod)
	if err != mockScheduler.err {
		t.Errorf("Expected %#v, Got %#v", mockScheduler.err, err)
	}
	if len(mockScheduler.pod.DesiredState.Manifest.Containers[0].ID) == 0 {
		t.Errorf("Expected container[0] to have ID set, Got %#v", mockScheduler.pod)
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
	body, err := api.Encode(&pod)
	expectNoError(t, err)
	podOut, err := storage.Extract(body)
	expectNoError(t, err)
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
