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

package memory

import (
	"errors"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// An implementation of PodRegistry and ControllerRegistry that is backed
// by memory. Mainly used for testing.
type Registry struct {
	podData        map[string]api.Pod
	controllerData map[string]api.ReplicationController
	serviceData    map[string]api.Service
}

// NewRegistry returns a new Registry.
func NewRegistry() *Registry {
	return &Registry{
		podData:        map[string]api.Pod{},
		controllerData: map[string]api.ReplicationController{},
		serviceData:    map[string]api.Service{},
	}
}

// CreateController registers the given replication controller.
func (r *Registry) CreateController(controller api.ReplicationController) error {
	r.controllerData[controller.ID] = controller
	return nil
}

// CreatePod registers the given pod.
func (r *Registry) CreatePod(machine string, pod api.Pod) error {
	r.podData[pod.ID] = pod
	return nil
}

// CreateService registers the given service.
func (r *Registry) CreateService(svc api.Service) error {
	r.serviceData[svc.ID] = svc
	return nil
}

// DeleteController deletes the named replication controller from the
// registry.
func (r *Registry) DeleteController(controllerID string) error {
	if _, ok := r.controllerData[controllerID]; !ok {
		return apiserver.NewNotFoundErr("replicationController", controllerID)
	}
	delete(r.controllerData, controllerID)
	return nil
}

// DeletePod deletes the named pod from the registry.
func (r *Registry) DeletePod(podID string) error {
	if _, ok := r.podData[podID]; !ok {
		return apiserver.NewNotFoundErr("pod", podID)
	}
	delete(r.podData, podID)
	return nil
}

// DeleteService deletes the named service from the registry.
// It returns an error if the service is not found in the registry.
func (r *Registry) DeleteService(name string) error {
	if _, ok := r.serviceData[name]; !ok {
		return apiserver.NewNotFoundErr("service", name)
	}
	delete(r.serviceData, name)
	return nil
}

// GetController returns an *api.ReplicationController for the name controller.
// It returns an error if the controller is not found in the registry.
func (r *Registry) GetController(controllerID string) (*api.ReplicationController, error) {
	controller, found := r.controllerData[controllerID]
	if found {
		return &controller, nil
	} else {
		return nil, apiserver.NewNotFoundErr("replicationController", controllerID)
	}
}

// GetPod returns an *api.Pod for the named pod.
// It returns an error if the pod is not found in the registry.
func (r *Registry) GetPod(podID string) (*api.Pod, error) {
	pod, found := r.podData[podID]
	if found {
		return &pod, nil
	} else {
		return nil, apiserver.NewNotFoundErr("pod", podID)
	}
}

// GetService returns an *api.Service for the named service.
// It returns an error if the service is not found in the registry.
func (r *Registry) GetService(name string) (*api.Service, error) {
	svc, found := r.serviceData[name]
	if !found {
		return nil, apiserver.NewNotFoundErr("service", name)
	}
	return &svc, nil
}

// ListControllers returns all registered replication controllers.
func (r *Registry) ListControllers() ([]api.ReplicationController, error) {
	result := []api.ReplicationController{}
	for _, value := range r.controllerData {
		result = append(result, value)
	}
	return result, nil
}

// ListPods returns all registered pods for the given selector.
func (r *Registry) ListPods(selector labels.Selector) ([]api.Pod, error) {
	result := []api.Pod{}
	for _, value := range r.podData {
		if selector.Matches(labels.Set(value.Labels)) {
			result = append(result, value)
		}
	}
	return result, nil
}

// ListServices returns all registered services.
func (r *Registry) ListServices() (api.ServiceList, error) {
	var list []api.Service
	for _, value := range r.serviceData {
		list = append(list, value)
	}
	return api.ServiceList{Items: list}, nil
}

// UpdateController updates the given controller in the registry.
// It returns an error if the controller is not found in the registry.
func (r *Registry) UpdateController(controller api.ReplicationController) error {
	if _, ok := r.controllerData[controller.ID]; !ok {
		return apiserver.NewNotFoundErr("replicationController", controller.ID)
	}
	r.controllerData[controller.ID] = controller
	return nil
}

// UpdateEndpoints always returns nil.
func (r *Registry) UpdateEndpoints(e api.Endpoints) error {
	return nil
}

// UpdatePod updates the given pod in the registry.
// It returns an error if the pod is not found in the registry.
func (r *Registry) UpdatePod(pod api.Pod) error {
	if _, ok := r.podData[pod.ID]; !ok {
		return apiserver.NewNotFoundErr("pod", pod.ID)
	}
	r.podData[pod.ID] = pod
	return nil
}

// UpdateService updates the given service in the registry.
// It returns an error if the service is not found in the registry.
func (r *Registry) UpdateService(svc api.Service) error {
	if _, ok := r.serviceData[svc.ID]; !ok {
		return apiserver.NewNotFoundErr("service", svc.ID)
	}
	return r.CreateService(svc)
}

// WatchControllers always returns an error.
// It is not implemented.
func (r *Registry) WatchControllers(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	return nil, errors.New("unimplemented")
}
