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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// rcStrategy implements verification logic for Replication Controllers.
type rcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = rcStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all Replication Controllers need to be within a namespace.
func (rcStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears the status of a replication controller before creation.
func (rcStrategy) ResetBeforeCreate(obj runtime.Object) {
	controller := obj.(*api.ReplicationController)
	controller.Status = api.ReplicationControllerStatus{}
}

// Validate validates a new replication controller.
func (rcStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	controller := obj.(*api.ReplicationController)
	return validation.ValidateReplicationController(controller)
}

// AllowCreateOnUpdate is false for replication controllers; this means a POST is
// needed to create one.
func (rcStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (rcStrategy) ValidateUpdate(obj, old runtime.Object) errors.ValidationErrorList {
	return validation.ValidateReplicationControllerUpdate(old.(*api.ReplicationController), obj.(*api.ReplicationController))
}

// PodLister is anything that knows how to list pods.
type PodLister interface {
	ListPods(ctx api.Context, labels labels.Selector) (*api.PodList, error)
}

// REST implements apiserver.RESTStorage for the replication controller service.
type REST struct {
	registry  Registry
	podLister PodLister
	strategy  rcStrategy
}

// NewREST returns a new apiserver.RESTStorage for the given registry and PodLister.
func NewREST(registry Registry, podLister PodLister) *REST {
	return &REST{
		registry:  registry,
		podLister: podLister,
		strategy:  Strategy,
	}
}

// Create registers the given ReplicationController with the system,
// which eventually leads to the controller manager acting on its behalf.
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, fmt.Errorf("not a replication controller: %#v", obj)
	}

	if err := rest.BeforeCreate(rs.strategy, ctx, obj); err != nil {
		return nil, err
	}

	out, err := rs.registry.CreateController(ctx, controller)
	if err != nil {
		err = rest.CheckGeneratedNameError(rs.strategy, err, controller)
	}
	return out, err
}

// Delete asynchronously deletes the ReplicationController specified by its id.
func (rs *REST) Delete(ctx api.Context, id string) (runtime.Object, error) {
	return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteController(ctx, id)
}

// Get obtains the ReplicationController specified by its id.
func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	controller, err := rs.registry.GetController(ctx, id)
	if err != nil {
		return nil, err
	}
	return controller, err
}

// List obtains a list of ReplicationControllers that match selector.
func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
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
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	controller, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, false, fmt.Errorf("not a replication controller: %#v", obj)
	}
	existingController, err := rs.registry.GetController(ctx, controller.Name)
	if err != nil {
		return nil, false, err
	}
	if err := rest.BeforeUpdate(rs.strategy, ctx, controller, existingController); err != nil {
		return nil, false, err
	}
	out, err := rs.registry.UpdateController(ctx, controller)
	return out, false, err
}

// Watch returns ReplicationController events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchControllers(ctx, label, field, resourceVersion)
}
