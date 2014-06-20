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

// Implementation of RESTStorage for the api server.
type ControllerRegistryStorage struct {
	registry ControllerRegistry
}

func MakeControllerRegistryStorage(registry ControllerRegistry) apiserver.RESTStorage {
	return &ControllerRegistryStorage{
		registry: registry,
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
	return apiserver.MakeAsync(func() interface{} { return api.Status{Status: api.StatusSuccess} }), storage.registry.DeleteController(id)
}

func (storage *ControllerRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := api.ReplicationController{}
	err := api.DecodeInto(body, &result)
	return result, err
}

func (storage *ControllerRegistryStorage) Create(controller interface{}) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() interface{} { return controller }), storage.registry.CreateController(controller.(api.ReplicationController))
}

func (storage *ControllerRegistryStorage) Update(controller interface{}) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() interface{} { return controller }), storage.registry.UpdateController(controller.(api.ReplicationController))
}
