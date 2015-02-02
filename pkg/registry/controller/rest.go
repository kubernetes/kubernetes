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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	rc "github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// PodLister is anything that knows how to list pods.
type PodLister interface {
	ListPods(ctx api.Context, labels labels.Selector) (*api.PodList, error)
}

// REST implements apiserver.RESTStorage for the replication controller service.
type REST struct {
	registry  Registry
	podLister PodLister
}

// NewREST returns a new apiserver.RESTStorage for the given registry and PodLister.
func NewREST(registry Registry, podLister PodLister) *REST {
	return &REST{
		registry:  registry,
		podLister: podLister,
	}
}

// Create registers the given ReplicationController.
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}

	if err := rest.BeforeCreate(rest.ReplicationControllers, ctx, obj); err != nil {
		return nil, err
	}

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := rs.registry.CreateController(ctx, controller); err != nil {
			err = rest.CheckGeneratedNameError(rest.ReplicationControllers, err, controller)
			return apiserver.RESTResult{}, err
		}
		return rs.registry.GetController(ctx, controller.Name)
	}), nil
}

// Delete asynchronously deletes the ReplicationController specified by its id.
func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteController(ctx, id)
	}), nil
}

// Get obtains the ReplicationController specified by its id.
func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	controller, err := rs.registry.GetController(ctx, id)
	if err != nil {
		return nil, err
	}
	rs.fillCurrentState(ctx, controller)
	return controller, err
}

// List obtains a list of ReplicationControllers that match selector.
func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	controllers, err := rs.registry.ListControllers(ctx)
	if err != nil {
		return nil, err
	}
	filtered := []api.ReplicationController{}
	for _, controller := range controllers.Items {
		if label.Matches(labels.Set(controller.Labels)) {
			rs.fillCurrentState(ctx, &controller)
			filtered = append(filtered, controller)
		}
	}
	controllers.Items = filtered
	return controllers, err
}

// New creates a new ReplicationController for use with Create and Update.
func (*REST) New() runtime.Object {
	return &api.ReplicationController{}
}

func (*REST) NewList() runtime.Object {
	return &api.ReplicationControllerList{}
}

// Update replaces a given ReplicationController instance with an existing
// instance in storage.registry.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}
	if !api.ValidNamespace(ctx, &controller.ObjectMeta) {
		return nil, errors.NewConflict("controller", controller.Namespace, fmt.Errorf("Controller.Namespace does not match the provided context"))
	}
	if errs := validation.ValidateReplicationController(controller); len(errs) > 0 {
		return nil, errors.NewInvalid("replicationController", controller.Name, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		err := rs.registry.UpdateController(ctx, controller)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetController(ctx, controller.Name)
	}), nil
}

// Watch returns ReplicationController events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchControllers(ctx, label, field, resourceVersion)
}

// TODO #2726: The controller should populate the current state, not the apiserver
func (rs *REST) fillCurrentState(ctx api.Context, controller *api.ReplicationController) error {
	if rs.podLister == nil {
		return nil
	}
	list, err := rs.podLister.ListPods(ctx, labels.Set(controller.Spec.Selector).AsSelector())
	if err != nil {
		return err
	}
	controller.Status.Replicas = len(rc.FilterActivePods(list.Items))
	return nil
}
