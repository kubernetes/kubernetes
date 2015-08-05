/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package limitrange

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

// REST provides the RESTStorage access patterns to work with LimitRange objects.
type REST struct {
	registry generic.Registry
}

// NewStorage returns a new REST. You must use a registry created by
// NewEtcdRegistry unless you're testing.
func NewStorage(registry generic.Registry) *REST {
	return &REST{
		registry: registry,
	}
}

// Create a LimitRange object
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	limitRange, ok := obj.(*api.LimitRange)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &limitRange.ObjectMeta) {
		return nil, errors.NewConflict("limitRange", limitRange.Namespace, fmt.Errorf("LimitRange.Namespace does not match the provided context"))
	}

	if len(limitRange.Name) == 0 {
		limitRange.Name = string(util.NewUUID())
	}

	if errs := validation.ValidateLimitRange(limitRange); len(errs) > 0 {
		return nil, errors.NewInvalid("limitRange", limitRange.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &limitRange.ObjectMeta)

	err := rs.registry.CreateWithName(ctx, limitRange.Name, limitRange)
	if err != nil {
		return nil, err
	}
	return rs.registry.Get(ctx, limitRange.Name)
}

// Update updates a LimitRange object.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	limitRange, ok := obj.(*api.LimitRange)
	if !ok {
		return nil, false, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &limitRange.ObjectMeta) {
		return nil, false, errors.NewConflict("limitRange", limitRange.Namespace, fmt.Errorf("LimitRange.Namespace does not match the provided context"))
	}

	oldObj, err := rs.registry.Get(ctx, limitRange.Name)
	if err != nil {
		return nil, false, err
	}

	editLimitRange := oldObj.(*api.LimitRange)

	// set the editable fields on the existing object
	editLimitRange.Labels = limitRange.Labels
	editLimitRange.ResourceVersion = limitRange.ResourceVersion
	editLimitRange.Annotations = limitRange.Annotations
	editLimitRange.Spec = limitRange.Spec

	if errs := validation.ValidateLimitRange(editLimitRange); len(errs) > 0 {
		return nil, false, errors.NewInvalid("limitRange", editLimitRange.Name, errs)
	}

	if err := rs.registry.UpdateWithName(ctx, editLimitRange.Name, editLimitRange); err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, editLimitRange.Name)
	return out, false, err
}

// Delete deletes the LimitRange with the specified name
func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.LimitRange)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return rs.registry.Delete(ctx, name, nil)
}

// Get gets a LimitRange with the specified name
func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	limitRange, ok := obj.(*api.LimitRange)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return limitRange, err
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels labels.Set, objFields fields.Set, err error) {
	return labels.Set{}, fields.Set{}, nil
}

func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return rs.registry.ListPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

func (rs *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}

// New returns a new api.LimitRange
func (*REST) New() runtime.Object {
	return &api.LimitRange{}
}

func (*REST) NewList() runtime.Object {
	return &api.LimitRangeList{}
}
