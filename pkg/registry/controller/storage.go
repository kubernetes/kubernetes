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

package controller

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"code.google.com/p/go-uuid/uuid"
)

// RegistryStorage stores data for the replication controller service.
// It implements apiserver.RESTStorage.
type RegistryStorage struct {
	registry    Registry
	podRegistry pod.Registry
	pollPeriod  time.Duration
}

// NewRegistryStorage returns a new apiserver.RESTStorage for the given
// registry and podRegistry.
func NewRegistryStorage(registry Registry, podRegistry pod.Registry) apiserver.RESTStorage {
	return &RegistryStorage{
		registry:    registry,
		podRegistry: podRegistry,
		pollPeriod:  time.Second * 10,
	}
}

// Create registers the given ReplicationController.
func (rs *RegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}
	if len(controller.ID) == 0 {
		controller.ID = uuid.NewUUID().String()
	}
	// Pod Manifest ID should be assigned by the pod API
	controller.DesiredState.PodTemplate.DesiredState.Manifest.ID = ""
	if errs := api.ValidateReplicationController(controller); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("replicationController", controller.ID, errs)
	}

	controller.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := rs.registry.CreateController(*controller)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetController(controller.ID)
	}), nil
}

// Delete asynchronously deletes the ReplicationController specified by its id.
func (rs *RegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteController(id)
	}), nil
}

// Get obtains the ReplicationController specified by its id.
func (rs *RegistryStorage) Get(id string) (interface{}, error) {
	controller, err := rs.registry.GetController(id)
	if err != nil {
		return nil, err
	}
	return controller, err
}

// List obtains a list of ReplicationControllers that match selector.
func (rs *RegistryStorage) List(selector labels.Selector) (interface{}, error) {
	controllers, err := rs.registry.ListControllers()
	if err != nil {
		return nil, err
	}
	filtered := []api.ReplicationController{}
	for _, controller := range controllers.Items {
		if selector.Matches(labels.Set(controller.Labels)) {
			filtered = append(filtered, controller)
		}
	}
	controllers.Items = filtered
	return controllers, err
}

// New creates a new ReplicationController for use with Create and Update.
func (rs RegistryStorage) New() interface{} {
	return &api.ReplicationController{}
}

// Update replaces a given ReplicationController instance with an existing
// instance in storage.registry.
func (rs *RegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}
	if errs := api.ValidateReplicationController(controller); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("replicationController", controller.ID, errs)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := rs.registry.UpdateController(*controller)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetController(controller.ID)
	}), nil
}

// Watch returns ReplicationController events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *RegistryStorage) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("no field selector implemented for controllers")
	}
	incoming, err := rs.registry.WatchControllers(resourceVersion)
	if err != nil {
		return nil, err
	}
	return watch.Filter(incoming, func(e watch.Event) (watch.Event, bool) {
		repController := e.Object.(*api.ReplicationController)
		return e, label.Matches(labels.Set(repController.Labels))
	}), nil
}

func (rs *RegistryStorage) waitForController(ctrl api.ReplicationController) (interface{}, error) {
	for {
		pods, err := rs.podRegistry.ListPods(labels.Set(ctrl.DesiredState.ReplicaSelector).AsSelector())
		if err != nil {
			return ctrl, err
		}
		if len(pods.Items) == ctrl.DesiredState.Replicas {
			break
		}
		time.Sleep(rs.pollPeriod)
	}
	return ctrl, nil
}
