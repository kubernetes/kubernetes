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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestListPodsEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	pods, err := registry.ListPods(labels.Everything())
	expectNoError(t, err)
	if len(pods) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemoryListPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreatePod("machine", api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
	pods, err := registry.ListPods(labels.Everything())
	expectNoError(t, err)
	if len(pods) != 1 || pods[0].ID != "foo" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestMemoryGetPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	pod, err := registry.GetPod("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetPod(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetPod(%q) = %v; expected failure with not found error", "foo", pod)
		}
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

func TestMemoryUpdatePods(t *testing.T) {
	registry := MakeMemoryRegistry()
	pod := api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.PodState{
			Host: "foo.com",
		},
	}
	err := registry.UpdatePod(pod)
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.UpdatePod(%q) failed with %v; expected failure with not found error", pod, err)
		} else {
			t.Errorf("registry.UpdatePod(%q) succeeded; expected failure with not found error", pod)
		}
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

func TestMemoryDeletePods(t *testing.T) {
	registry := MakeMemoryRegistry()
	err := registry.DeletePod("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.DeletePod(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.DeletePod(%q) succeeded; expected failure with not found error", "foo")
		}
	}
}

func TestMemorySetDeleteGetPods(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedPod := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreatePod("machine", expectedPod)
	registry.DeletePod("foo")
	pod, err := registry.GetPod("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetPod(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetPod(%q) = %v; expected failure with not found error", "foo", pod)
		}
	}
}

func TestListControllersEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	ctls, err := registry.ListControllers()
	expectNoError(t, err)
	if len(ctls) != 0 {
		t.Errorf("Unexpected controller list: %#v", ctls)
	}
}

func TestMemoryListControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreateController(api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}})
	ctls, err := registry.ListControllers()
	expectNoError(t, err)
	if len(ctls) != 1 || ctls[0].ID != "foo" {
		t.Errorf("Unexpected controller list: %#v", ctls)
	}
}

func TestMemoryGetController(t *testing.T) {
	registry := MakeMemoryRegistry()
	ctl, err := registry.GetController("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetController(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetController(%q) = %v; expected failure with not found error", "foo", ctl)
		}
	}
}

func TestMemorySetGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	ctl, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != ctl.ID {
		t.Errorf("Unexpected controller, expected %#v, actual %#v", expectedController, ctl)
	}
}

func TestMemoryUpdateController(t *testing.T) {
	registry := MakeMemoryRegistry()
	ctl := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
	}
	err := registry.UpdateController(ctl)
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.UpdateController(%q) failed with %v; expected failure with not found error", ctl, err)
		} else {
			t.Errorf("registry.UpdateController(%q) succeeded; expected failure with not found error", ctl)
		}
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
	ctl, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != ctl.ID || ctl.DesiredState.Replicas != expectedController.DesiredState.Replicas {
		t.Errorf("Unexpected controller, expected %#v, actual %#v", expectedController, ctl)
	}
}

func TestMemoryDeleteController(t *testing.T) {
	registry := MakeMemoryRegistry()
	err := registry.DeleteController("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.DeleteController(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.DeleteController(%q) succeeded; expected failure with not found error", "foo")
		}
	}
}

func TestMemorySetDeleteGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	registry.DeleteController("foo")
	ctl, err := registry.GetController("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetController(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetController(%q) = %v; expected failure with not found error", "foo", ctl)
		}
	}
}

func TestListServicesEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	svcs, err := registry.ListServices()
	expectNoError(t, err)
	if len(svcs.Items) != 0 {
		t.Errorf("Unexpected service list: %#v", svcs)
	}
}

func TestMemoryListServices(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreateService(api.Service{JSONBase: api.JSONBase{ID: "foo"}})
	svcs, err := registry.ListServices()
	expectNoError(t, err)
	if len(svcs.Items) != 1 || svcs.Items[0].ID != "foo" {
		t.Errorf("Unexpected service list: %#v", svcs)
	}
}

func TestMemoryGetService(t *testing.T) {
	registry := MakeMemoryRegistry()
	svc, err := registry.GetService("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetService(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetService(%q) = %v; expected failure with not found error", "foo", svc)
		}
	}
}

func TestMemorySetGetServices(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedService := api.Service{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateService(expectedService)
	svc, err := registry.GetService("foo")
	expectNoError(t, err)
	if expectedService.ID != svc.ID {
		t.Errorf("Unexpected service, expected %#v, actual %#v", expectedService, svc)
	}
}

func TestMemoryUpdateService(t *testing.T) {
	registry := MakeMemoryRegistry()
	svc := api.Service{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		Port: 9000,
	}
	err := registry.UpdateService(svc)
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.UpdateService(%q) failed with %v; expected failure with not found error", svc, err)
		} else {
			t.Errorf("registry.UpdateService(%q) succeeded; expected failure with not found error", svc)
		}
	}
}

func TestMemorySetUpdateGetServices(t *testing.T) {
	registry := MakeMemoryRegistry()
	oldService := api.Service{JSONBase: api.JSONBase{ID: "foo"}}
	expectedService := api.Service{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		Port: 9000,
	}
	registry.CreateService(oldService)
	registry.UpdateService(expectedService)
	svc, err := registry.GetService("foo")
	expectNoError(t, err)
	if expectedService.ID != svc.ID || svc.Port != expectedService.Port {
		t.Errorf("Unexpected service, expected %#v, actual %#v", expectedService, svc)
	}
}

func TestMemoryDeleteService(t *testing.T) {
	registry := MakeMemoryRegistry()
	err := registry.DeleteService("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.DeleteService(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.DeleteService(%q) succeeded; expected failure with not found error", "foo")
		}
	}
}

func TestMemorySetDeleteGetServices(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedService := api.Service{JSONBase: api.JSONBase{ID: "foo"}}
	registry.CreateService(expectedService)
	registry.DeleteService("foo")
	svc, err := registry.GetService("foo")
	if !apiserver.IsNotFound(err) {
		if err != nil {
			t.Errorf("registry.GetService(%q) failed with %v; expected failure with not found error", "foo", err)
		} else {
			t.Errorf("registry.GetService(%q) = %v; expected failure with not found error", "foo", svc)
		}
	}
}
