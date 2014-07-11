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
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/fsouza/go-dockerclient"
)

func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

func expectApiStatusError(t *testing.T, ch <-chan interface{}, msg string) {
	out := <-ch
	status, ok := out.(*api.Status)
	if !ok {
		t.Errorf("Expected an api.Status object, was %#v", out)
		return
	}
	if msg != status.Details {
		t.Errorf("Expected %#v, was %s", msg, status.Details)
	}
}

func expectPod(t *testing.T, ch <-chan interface{}) (*api.Pod, bool) {
	out := <-ch
	pod, ok := out.(*api.Pod)
	if !ok || pod == nil {
		t.Errorf("Expected an api.Pod object, was %#v", out)
		return nil, false
	}
	return pod, true
}

func TestCreatePodRegistryError(t *testing.T) {
	mockRegistry := &MockPodRegistry{
		err: fmt.Errorf("test error"),
	}
	storage := PodRegistryStorage{
		scheduler: &MockScheduler{},
		registry:  mockRegistry,
	}
	pod := api.Pod{}
	ch, err := storage.Create(pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, mockRegistry.err.Error())
}

type MockScheduler struct {
	err     error
	pod     api.Pod
	machine string
}

func (m *MockScheduler) Schedule(pod api.Pod, lister scheduler.MinionLister) (string, error) {
	m.pod = pod
	return m.machine, m.err
}

func TestCreatePodSchedulerError(t *testing.T) {
	mockScheduler := MockScheduler{
		err: fmt.Errorf("test error"),
	}
	storage := PodRegistryStorage{
		scheduler: &mockScheduler,
	}
	pod := api.Pod{}
	ch, err := storage.Create(pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, mockScheduler.err.Error())
}

type MockPodStorageRegistry struct {
	MockPodRegistry
	machine string
}

func (r *MockPodStorageRegistry) CreatePod(machine string, pod api.Pod) error {
	r.MockPodRegistry.pod = &pod
	r.machine = machine
	return r.MockPodRegistry.err
}

func TestCreatePodSetsIds(t *testing.T) {
	mockRegistry := &MockPodStorageRegistry{
		MockPodRegistry: MockPodRegistry{err: fmt.Errorf("test error")},
	}
	storage := PodRegistryStorage{
		scheduler: &MockScheduler{machine: "test"},
		registry:  mockRegistry,
	}
	pod := api.Pod{}
	ch, err := storage.Create(pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, mockRegistry.err.Error())

	if len(mockRegistry.MockPodRegistry.pod.ID) == 0 {
		t.Errorf("Expected pod ID to be set, Got %#v", pod)
	}
	if mockRegistry.MockPodRegistry.pod.DesiredState.Manifest.ID != mockRegistry.MockPodRegistry.pod.ID {
		t.Errorf("Expected manifest ID to be equal to pod ID, Got %#v", pod)
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
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Containers: []api.Container{
				{Name: "containerA"},
				{Name: "containerB"},
			},
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	status := makePodStatus(pod)
	if status != api.PodPending {
		t.Errorf("Expected 'Pending', got '%s'", status)
	}

	runningState := docker.Container{
		State: docker.State{
			Running: true,
		},
	}
	stoppedState := docker.Container{
		State: docker.State{
			Running: false,
		},
	}

	// All running.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": runningState,
				"containerB": runningState,
			},
		},
	}
	status = makePodStatus(pod)
	if status != api.PodRunning {
		t.Errorf("Expected 'Running', got '%s'", status)
	}

	// All stopped.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": stoppedState,
				"containerB": stoppedState,
			},
		},
	}
	status = makePodStatus(pod)
	if status != api.PodStopped {
		t.Errorf("Expected 'Stopped', got '%s'", status)
	}

	// Mixed state.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": runningState,
				"containerB": stoppedState,
			},
		},
	}
	status = makePodStatus(pod)
	if status != api.PodPending {
		t.Errorf("Expected 'Pending', got '%s'", status)
	}

	// Mixed state.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": runningState,
			},
		},
	}
	status = makePodStatus(pod)
	if status != api.PodPending {
		t.Errorf("Expected 'Pending', got '%s'", status)
	}
}

func TestCreatePod(t *testing.T) {
	mockRegistry := MockPodRegistry{
		pod: &api.Pod{
			JSONBase: api.JSONBase{ID: "foo"},
			CurrentState: api.PodState{
				Status: api.PodPending,
			},
		},
	}
	storage := PodRegistryStorage{
		registry:      &mockRegistry,
		podPollPeriod: time.Millisecond * 100,
		scheduler:     scheduler.MakeRoundRobinScheduler(),
		minionLister:  MakeMinionRegistry([]string{"machine"}),
	}
	pod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	channel, err := storage.Create(pod)
	expectNoError(t, err)
	select {
	case <-time.After(time.Millisecond * 100):
		// Do nothing, this is expected.
	case <-channel:
		t.Error("Unexpected read from async channel")
	}
	mockRegistry.UpdatePod(api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
		CurrentState: api.PodState{
			Status: api.PodRunning,
		},
	})
	select {
	case <-time.After(time.Second * 1):
		t.Error("Unexpected timeout")
	case <-channel:
		// Do nothing, this is expected.
	}
}

type FakePodInfoGetter struct {
	info api.PodInfo
	err  error
}

func (f *FakePodInfoGetter) GetPodInfo(host, podID string) (api.PodInfo, error) {
	return f.info, f.err
}

func TestFillPodInfo(t *testing.T) {
	expectedIP := "1.2.3.4"
	fakeGetter := FakePodInfoGetter{
		info: map[string]docker.Container{
			"net": {
				ID:   "foobar",
				Path: "bin/run.sh",
				NetworkSettings: &docker.NetworkSettings{
					IPAddress: expectedIP,
				},
			},
		},
	}
	storage := PodRegistryStorage{
		podCache: &fakeGetter,
	}

	pod := api.Pod{}

	storage.fillPodInfo(&pod)

	if !reflect.DeepEqual(fakeGetter.info, pod.CurrentState.Info) {
		t.Errorf("Expected: %#v, Got %#v", fakeGetter.info, pod.CurrentState.Info)
	}

	if pod.CurrentState.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, pod.CurrentState.PodIP)
	}
}

func TestFillPodInfoNoData(t *testing.T) {
	expectedIP := ""
	fakeGetter := FakePodInfoGetter{
		info: map[string]docker.Container{
			"net": {
				ID:   "foobar",
				Path: "bin/run.sh",
			},
		},
	}
	storage := PodRegistryStorage{
		podCache: &fakeGetter,
	}

	pod := api.Pod{}

	storage.fillPodInfo(&pod)

	if !reflect.DeepEqual(fakeGetter.info, pod.CurrentState.Info) {
		t.Errorf("Expected %#v, Got %#v", fakeGetter.info, pod.CurrentState.Info)
	}

	if pod.CurrentState.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, pod.CurrentState.PodIP)
	}
}
