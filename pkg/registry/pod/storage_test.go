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

package pod

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"

	"github.com/fsouza/go-dockerclient"
)

func expectApiStatusError(t *testing.T, ch <-chan interface{}, msg string) {
	out := <-ch
	status, ok := out.(*api.Status)
	if !ok {
		t.Errorf("Expected an api.Status object, was %#v", out)
		return
	}
	if msg != status.Message {
		t.Errorf("Expected %#v, was %s", msg, status.Message)
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
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := RegistryStorage{
		registry: podRegistry,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	ch, err := storage.Create(pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())
}

func TestCreatePodSetsIds(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := RegistryStorage{
		registry: podRegistry,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	ch, err := storage.Create(pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())

	if len(podRegistry.Pod.ID) == 0 {
		t.Errorf("Expected pod ID to be set, Got %#v", pod)
	}
	if podRegistry.Pod.DesiredState.Manifest.ID != podRegistry.Pod.ID {
		t.Errorf("Expected manifest ID to be equal to pod ID, Got %#v", pod)
	}
}

func TestListPodsError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := RegistryStorage{
		registry: podRegistry,
	}
	pods, err := storage.List(labels.Everything())
	if err != podRegistry.Err {
		t.Errorf("Expected %#v, Got %#v", podRegistry.Err, err)
	}
	if pods.(*api.PodList) != nil {
		t.Errorf("Unexpected non-nil pod list: %#v", pods)
	}
}

func TestListEmptyPodList(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(&api.PodList{JSONBase: api.JSONBase{ResourceVersion: 1}})
	storage := RegistryStorage{
		registry: podRegistry,
	}
	pods, err := storage.List(labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.(*api.PodList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", pods)
	}
	if pods.(*api.PodList).ResourceVersion != 1 {
		t.Errorf("Unexpected resource version: %#v", pods)
	}
}

func TestListPodList(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pods = &api.PodList{
		Items: []api.Pod{
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
	storage := RegistryStorage{
		registry: podRegistry,
	}
	podsObj, err := storage.List(labels.Everything())
	pods := podsObj.(*api.PodList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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

func TestPodDecode(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	storage := RegistryStorage{
		registry: podRegistry,
	}
	expected := &api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	}
	body, err := api.Encode(expected)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := api.DecodeInto(body, actual); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}

func TestGetPod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	storage := RegistryStorage{
		registry: podRegistry,
	}
	obj, err := storage.Get("foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := podRegistry.Pod, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}
}

func TestGetPodCloud(t *testing.T) {
	fakeCloud := &fake_cloud.FakeCloud{}
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	storage := RegistryStorage{
		registry:      podRegistry,
		cloudProvider: fakeCloud,
	}
	obj, err := storage.Get("foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := podRegistry.Pod, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}
	if len(fakeCloud.Calls) != 1 || fakeCloud.Calls[0] != "ip-address" {
		t.Errorf("Unexpected calls: %#v", fakeCloud.Calls)
	}
}

func TestMakePodStatus(t *testing.T) {
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
			Containers: []api.Container{
				{Name: "containerA"},
				{Name: "containerB"},
			},
		},
	}
	currentState := api.PodState{
		Host: "machine",
	}
	pod := &api.Pod{DesiredState: desiredState, CurrentState: currentState}
	status := getPodStatus(pod)
	if status != api.PodWaiting {
		t.Errorf("Expected 'Waiting', got '%s'", status)
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
			Host: "machine",
		},
	}
	status = getPodStatus(pod)
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
			Host: "machine",
		},
	}
	status = getPodStatus(pod)
	if status != api.PodTerminated {
		t.Errorf("Expected 'Terminated', got '%s'", status)
	}

	// Mixed state.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": runningState,
				"containerB": stoppedState,
			},
			Host: "machine",
		},
	}
	status = getPodStatus(pod)
	if status != api.PodWaiting {
		t.Errorf("Expected 'Waiting', got '%s'", status)
	}

	// Mixed state.
	pod = &api.Pod{
		DesiredState: desiredState,
		CurrentState: api.PodState{
			Info: map[string]docker.Container{
				"containerA": runningState,
			},
			Host: "machine",
		},
	}
	status = getPodStatus(pod)
	if status != api.PodWaiting {
		t.Errorf("Expected 'Waiting', got '%s'", status)
	}
}

func TestPodStorageValidatesCreate(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := RegistryStorage{
		registry: podRegistry,
	}
	pod := &api.Pod{}
	c, err := storage.Create(pod)
	if c != nil {
		t.Errorf("Expected nil channel")
	}
	if !apiserver.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

func TestPodStorageValidatesUpdate(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := RegistryStorage{
		registry: podRegistry,
	}
	pod := &api.Pod{}
	c, err := storage.Update(pod)
	if c != nil {
		t.Errorf("Expected nil channel")
	}
	if !apiserver.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

func TestCreatePod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
		CurrentState: api.PodState{
			Host: "machine",
		},
	}
	storage := RegistryStorage{
		registry:      podRegistry,
		podPollPeriod: time.Millisecond * 100,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{
		JSONBase:     api.JSONBase{ID: "foo"},
		DesiredState: desiredState,
	}
	channel, err := storage.Create(pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-channel:
		// Do nothing, this is expected.
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout on async channel")
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
	storage := RegistryStorage{
		podCache: &fakeGetter,
	}
	pod := api.Pod{DesiredState: api.PodState{Host: "foo"}}
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
	storage := RegistryStorage{
		podCache: &fakeGetter,
	}
	pod := api.Pod{DesiredState: api.PodState{Host: "foo"}}
	storage.fillPodInfo(&pod)
	if !reflect.DeepEqual(fakeGetter.info, pod.CurrentState.Info) {
		t.Errorf("Expected %#v, Got %#v", fakeGetter.info, pod.CurrentState.Info)
	}
	if pod.CurrentState.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, pod.CurrentState.PodIP)
	}
}
