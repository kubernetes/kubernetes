/*
Copyright 2015 Google Inc. All rights reserved.

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

package autoscaler

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST provides the RESTStorage access patterns to work with AutoScaler objects.
type REST struct {
	registry generic.Registry
}

// NewREST returns a new REST. You must use a registry created by
// NewEtcdRegistry unless you're testing.
func NewREST(registry generic.Registry) *REST {
	return &REST{
		registry: registry,
	}
}

// New returns a new AutoScaler.
func (*REST) New() runtime.Object {
	return &api.AutoScaler{}
}

// NewList returns a new AutoScaler.
func (*REST) NewList() runtime.Object {
	return &api.AutoScalerList{}
}

// List selects resources in the storage which match to the selector.
func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return rs.registry.ListPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

// getAttrs is passed to the predicate functions
func (rs *REST) getAttrs(obj runtime.Object) (objLabels labels.Set, objFields fields.Set, err error) {
	autoScaler, ok := obj.(*api.AutoScaler)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type")
	}
	return labels.Set(autoScaler.Labels), fields.Set{}, nil
}

// Get finds an AutoScaler in the storage by id and returns it.
func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	autoScaler, ok := obj.(*api.AutoScaler)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return autoScaler, err
}

// Delete finds an AutoScaler in the storage and deletes it.
func (rs *REST) Delete(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.AutoScaler)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return rs.registry.Delete(ctx, id)
}

// Create creates a new AutoScaler.
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	if err := rest.BeforeCreate(rest.AutoScalers, ctx, obj); err != nil {
		return nil, err
	}

	//should never fail here, BeforeCreate does a cast to do validation
	autoScaler := obj.(*api.AutoScaler)
	if err := rs.registry.CreateWithName(ctx, autoScaler.Name, autoScaler); err != nil {
		return nil, err
	}

	return rs.registry.Get(ctx, autoScaler.Name)
}

// Update updates an AutoScaler object.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	autoScaler, ok := obj.(*api.AutoScaler)
	if !ok {
		return nil, false, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &autoScaler.ObjectMeta) {
		return nil, false, errors.NewConflict("autoscaler", autoScaler.Namespace, fmt.Errorf("AutoScaler.Namespace does not match the provided context"))
	}

	oldObj, err := rs.registry.Get(ctx, autoScaler.Name)
	if err != nil {
		return nil, false, err
	}

	editAutoScaler := oldObj.(*api.AutoScaler)
	if errs := validation.ValidateAutoScalerUpdate(editAutoScaler, autoScaler); len(errs) > 0 {
		return nil, false, errors.NewInvalid("autoscaler", editAutoScaler.Name, errs)
	}

	// passed update validation (ensures immutable meta is not trying to be changed),
	// now copy over the relevant fields that can be updated
	editAutoScaler.Spec = autoScaler.Spec

	err = rs.registry.UpdateWithName(ctx, editAutoScaler.Name, editAutoScaler)
	if err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, editAutoScaler.Name)
	return out, false, err
}

// Watch provides the ability to watch auto-scalers for changes
func (rs *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}
