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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// An implementation of PodRegistry and ControllerRegistry that is backed by memory
// Mainly used for testing.
type MemoryRegistry struct {
	podData        map[string]api.Pod
	controllerData map[string]api.ReplicationController
	serviceData    map[string]api.Service
}

func MakeMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		podData:        map[string]api.Pod{},
		controllerData: map[string]api.ReplicationController{},
		serviceData:    map[string]api.Service{},
	}
}

func (registry *MemoryRegistry) ListPods(selector labels.Selector) ([]api.Pod, error) {
	result := []api.Pod{}
	for _, value := range registry.podData {
		if selector.Matches(labels.Set(value.Labels)) {
			result = append(result, value)
		}
	}
	return result, nil
}

func (registry *MemoryRegistry) GetPod(podID string) (*api.Pod, error) {
	pod, found := registry.podData[podID]
	if found {
		return &pod, nil
	} else {
		return nil, apiserver.NewNotFoundErr("pod", podID)
	}
}

func (registry *MemoryRegistry) CreatePod(machine string, pod api.Pod) error {
	registry.podData[pod.ID] = pod
	return nil
}

func (registry *MemoryRegistry) DeletePod(podID string) error {
	if _, ok := registry.podData[podID]; !ok {
		return apiserver.NewNotFoundErr("pod", podID)
	}
	delete(registry.podData, podID)
	return nil
}

func (registry *MemoryRegistry) UpdatePod(pod api.Pod) error {
	if _, ok := registry.podData[pod.ID]; !ok {
		return apiserver.NewNotFoundErr("pod", pod.ID)
	}
	registry.podData[pod.ID] = pod
	return nil
}

func (registry *MemoryRegistry) ListControllers() ([]api.ReplicationController, error) {
	result := []api.ReplicationController{}
	for _, value := range registry.controllerData {
		result = append(result, value)
	}
	return result, nil
}

func (registry *MemoryRegistry) GetController(controllerID string) (*api.ReplicationController, error) {
	controller, found := registry.controllerData[controllerID]
	if found {
		return &controller, nil
	} else {
		return nil, apiserver.NewNotFoundErr("replicationController", controllerID)
	}
}

func (registry *MemoryRegistry) CreateController(controller api.ReplicationController) error {
	registry.controllerData[controller.ID] = controller
	return nil
}

func (registry *MemoryRegistry) DeleteController(controllerID string) error {
	if _, ok := registry.controllerData[controllerID]; !ok {
		return apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	delete(registry.controllerData, controllerID)
	return nil
}

func (registry *MemoryRegistry) UpdateController(controller api.ReplicationController) error {
	if _, ok := registry.controllerData[controller.ID]; !ok {
		return apiserver.NewNotFoundErr("replicationController", controller.ID)
	}
	registry.controllerData[controller.ID] = controller
	return nil
}

func (registry *MemoryRegistry) ListServices() (api.ServiceList, error) {
	var list []api.Service
	for _, value := range registry.serviceData {
		list = append(list, value)
	}
	return api.ServiceList{Items: list}, nil
}

func (registry *MemoryRegistry) CreateService(svc api.Service) error {
	registry.serviceData[svc.ID] = svc
	return nil
}

func (registry *MemoryRegistry) GetService(name string) (*api.Service, error) {
	svc, found := registry.serviceData[name]
	if found {
		return &svc, nil
	} else {
		return nil, apiserver.NewNotFoundErr("service", name)
	}
}

func (registry *MemoryRegistry) DeleteService(name string) error {
	if _, ok := registry.serviceData[name]; !ok {
		return apiserver.NewNotFoundErr("service", name)
	}
	delete(registry.serviceData, name)
	return nil
}

func (registry *MemoryRegistry) UpdateService(svc api.Service) error {
	if _, ok := registry.serviceData[svc.ID]; !ok {
		return apiserver.NewNotFoundErr("service", svc.ID)
	}
	return registry.CreateService(svc)
}

func (registry *MemoryRegistry) UpdateEndpoints(e api.Endpoints) error {
	return nil
}
