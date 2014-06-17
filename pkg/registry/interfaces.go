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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// PodRegistry is an interface implemented by things that know how to store Pod objects.
type PodRegistry interface {
	// ListPods obtains a list of pods that match query.
	ListPods(query labels.Query) ([]api.Pod, error)
	// Get a specific pod
	GetPod(podID string) (*api.Pod, error)
	// Create a pod based on a specification, schedule it onto a specific machine.
	CreatePod(machine string, pod api.Pod) error
	// Update an existing pod
	UpdatePod(pod api.Pod) error
	// Delete an existing pod
	DeletePod(podID string) error
}

// ControllerRegistry is an interface for things that know how to store Controllers.
type ControllerRegistry interface {
	ListControllers() ([]api.ReplicationController, error)
	GetController(controllerId string) (*api.ReplicationController, error)
	CreateController(controller api.ReplicationController) error
	UpdateController(controller api.ReplicationController) error
	DeleteController(controllerId string) error
}

// ServiceRegistry is an interface for things that know how to store services.
type ServiceRegistry interface {
	ListServices() (api.ServiceList, error)
	CreateService(svc api.Service) error
	GetService(name string) (*api.Service, error)
	DeleteService(name string) error
	UpdateService(svc api.Service) error
	UpdateEndpoints(e api.Endpoints) error
}
