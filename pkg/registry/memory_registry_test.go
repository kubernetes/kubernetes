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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestListPodsEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	pods, err := registry.ListPods(nil)
	expectNoError(t, err)
	if len(pods) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemoryListPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreatePod("machine", api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
	pods, err := registry.ListPods(nil)
	expectNoError(t, err)
	if len(pods) != 1 || pods[0].ID != "foo" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemorySetGetPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedPod := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreatePod("machine", expectedPod)
	pod, err := registry.GetPod("foo")
	expectNoError(t, err)
	if expectedPod.ID != pod.ID {
		t.Errorf("Unexpected pod, expected %#v, actual %#v", expectedPod, pod)
	}
}

func TestMemorySetUpdateGetPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	oldPod := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	expectedPod := api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.PodState{
			Host: "foo.com",
		},
	}
	registry.CreatePod("machine", oldPod)
	registry.UpdatePod(expectedPod)
	pod, err := registry.GetPod("foo")
	expectNoError(t, err)
	if expectedPod.ID != pod.ID || pod.DesiredState.Host != expectedPod.DesiredState.Host {
		t.Errorf("Unexpected pod, expected %#v, actual %#v", expectedPod, pod)
	}
}

func TestMemorySetDeleteGetPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedPod := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreatePod("machine", expectedPod)
	registry.DeletePod("foo")
	pod, err := registry.GetPod("foo")
	expectNoError(t, err)
	if pod != nil {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}

func TestListControllersEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	pods, err := registry.ListControllers()
	expectNoError(t, err)
	if len(pods) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemoryListControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreateController(api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}})
	pods, err := registry.ListControllers()
	expectNoError(t, err)
	if len(pods) != 1 || pods[0].ID != "foo" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemorySetGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	pod, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != pod.ID {
		t.Errorf("Unexpected pod, expected %#v, actual %#v", expectedController, pod)
	}
}

func TestMemorySetUpdateGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	oldController := api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}
	expectedController := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
	}
	registry.CreateController(oldController)
	registry.UpdateController(expectedController)
	pod, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != pod.ID || pod.DesiredState.Replicas != expectedController.DesiredState.Replicas {
		t.Errorf("Unexpected pod, expected %#v, actual %#v", expectedController, pod)
	}
}

func TestMemorySetDeleteGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	registry.DeleteController("foo")
	pod, err := registry.GetController("foo")
	expectNoError(t, err)
	if pod != nil {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}
