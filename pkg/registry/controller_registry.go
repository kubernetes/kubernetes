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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Implementation of RESTStorage for the api server.
type ControllerRegistryStorage struct {
	registry    ControllerRegistry
	podRegistry PodRegistry
	// Period in between polls when waiting for a controller to complete
	pollPeriod time.Duration
}

func MakeControllerRegistryStorage(registry ControllerRegistry, podRegistry PodRegistry) apiserver.RESTStorage {
	return &ControllerRegistryStorage{
		registry:    registry,
		podRegistry: podRegistry,
		pollPeriod:  time.Second * 10,
	}
}

func (storage *ControllerRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := api.ReplicationControllerList{}
	controllers, err := storage.registry.ListControllers()
	if err == nil {
		for _, controller := range controllers {
			if selector.Matches(labels.Set(controller.Labels)) {
				result.Items = append(result.Items, controller)
			}
		}
	}
	return result, err
}

func (storage *ControllerRegistryStorage) Get(id string) (interface{}, error) {
	controller, err := storage.registry.GetController(id)
	if err != nil {
		return nil, err
	}
	return controller, err
}

func (storage *ControllerRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, storage.registry.DeleteController(id)
	}), nil
}

func (storage *ControllerRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := api.ReplicationController{}
	err := api.DecodeInto(body, &result)
	return result, err
}

func (storage *ControllerRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	controller, ok := obj.(api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}
	if controller.ID == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", controller)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.CreateController(controller)
		if err != nil {
			return nil, err
		}
		return storage.waitForController(controller)
	}), nil
}

func (storage *ControllerRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	controller, ok := obj.(api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}
	if controller.ID == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", controller)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.UpdateController(controller)
		if err != nil {
			return nil, err
		}
		return storage.waitForController(controller)
	}), nil
}

func (storage *ControllerRegistryStorage) waitForController(ctrl api.ReplicationController) (interface{}, error) {
	for {
		pods, err := storage.podRegistry.ListPods(labels.Set(ctrl.DesiredState.ReplicaSelector).AsSelector())
		if err != nil {
			return ctrl, err
		}
		if len(pods) == ctrl.DesiredState.Replicas {
			break
		}
		time.Sleep(storage.pollPeriod)
	}
	return ctrl, nil
}
