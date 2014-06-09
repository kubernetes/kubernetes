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
	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// An implementation of TaskRegistry and ControllerRegistry that is backed by memory
// Mainly used for testing.
type MemoryRegistry struct {
	taskData       map[string]Pod
	controllerData map[string]ReplicationController
	serviceData    map[string]Service
}

func MakeMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		taskData:       map[string]Pod{},
		controllerData: map[string]ReplicationController{},
		serviceData:    map[string]Service{},
	}
}

func (registry *MemoryRegistry) ListTasks(labelQuery *map[string]string) ([]Pod, error) {
	result := []Pod{}
	for _, value := range registry.taskData {
		if LabelsMatch(value, labelQuery) {
			result = append(result, value)
		}
	}
	return result, nil
}

func (registry *MemoryRegistry) GetTask(taskID string) (*Pod, error) {
	task, found := registry.taskData[taskID]
	if found {
		return &task, nil
	} else {
		return nil, nil
	}
}

func (registry *MemoryRegistry) CreateTask(machine string, task Pod) error {
	registry.taskData[task.ID] = task
	return nil
}

func (registry *MemoryRegistry) DeleteTask(taskID string) error {
	delete(registry.taskData, taskID)
	return nil
}

func (registry *MemoryRegistry) UpdateTask(task Pod) error {
	registry.taskData[task.ID] = task
	return nil
}

func (registry *MemoryRegistry) ListControllers() ([]ReplicationController, error) {
	result := []ReplicationController{}
	for _, value := range registry.controllerData {
		result = append(result, value)
	}
	return result, nil
}

func (registry *MemoryRegistry) GetController(controllerID string) (*ReplicationController, error) {
	controller, found := registry.controllerData[controllerID]
	if found {
		return &controller, nil
	} else {
		return nil, nil
	}
}

func (registry *MemoryRegistry) CreateController(controller ReplicationController) error {
	registry.controllerData[controller.ID] = controller
	return nil
}

func (registry *MemoryRegistry) DeleteController(controllerId string) error {
	delete(registry.controllerData, controllerId)
	return nil
}

func (registry *MemoryRegistry) UpdateController(controller ReplicationController) error {
	registry.controllerData[controller.ID] = controller
	return nil
}

func (registry *MemoryRegistry) ListServices() (ServiceList, error) {
	var list []Service
	for _, value := range registry.serviceData {
		list = append(list, value)
	}
	return ServiceList{Items: list}, nil
}

func (registry *MemoryRegistry) CreateService(svc Service) error {
	registry.serviceData[svc.ID] = svc
	return nil
}

func (registry *MemoryRegistry) GetService(name string) (*Service, error) {
	svc, found := registry.serviceData[name]
	if found {
		return &svc, nil
	} else {
		return nil, nil
	}
}

func (registry *MemoryRegistry) DeleteService(name string) error {
	delete(registry.serviceData, name)
	return nil
}

func (registry *MemoryRegistry) UpdateService(svc Service) error {
	return registry.CreateService(svc)
}

func (registry *MemoryRegistry) UpdateEndpoints(e Endpoints) error {
	return nil
}
