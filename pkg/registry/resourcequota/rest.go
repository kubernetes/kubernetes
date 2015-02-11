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

package resourcequota

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST provides the RESTStorage access patterns to work with ResourceQuota objects.
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

// Create a ResourceQuota object
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	resourceQuota, ok := obj.(*api.ResourceQuota)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &resourceQuota.ObjectMeta) {
		return nil, errors.NewConflict("resourceQuota", resourceQuota.Namespace, fmt.Errorf("ResourceQuota.Namespace does not match the provided context"))
	}

	if len(resourceQuota.Name) == 0 {
		resourceQuota.Name = string(util.NewUUID())
	}

	// callers are not able to set status, instead, it is supplied via a control loop
	resourceQuota.Status = api.ResourceQuotaStatus{}

	if errs := validation.ValidateResourceQuota(resourceQuota); len(errs) > 0 {
		return nil, errors.NewInvalid("resourceQuota", resourceQuota.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &resourceQuota.ObjectMeta)

	err := rs.registry.CreateWithName(ctx, resourceQuota.Name, resourceQuota)
	if err != nil {
		return nil, err
	}
	return rs.registry.Get(ctx, resourceQuota.Name)
}

// Update updates a ResourceQuota object.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	resourceQuota, ok := obj.(*api.ResourceQuota)
	if !ok {
		return nil, false, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &resourceQuota.ObjectMeta) {
		return nil, false, errors.NewConflict("resourceQuota", resourceQuota.Namespace, fmt.Errorf("ResourceQuota.Namespace does not match the provided context"))
	}

	oldObj, err := rs.registry.Get(ctx, resourceQuota.Name)
	if err != nil {
		return nil, false, err
	}

	editResourceQuota := oldObj.(*api.ResourceQuota)

	// set the editable fields on the existing object
	editResourceQuota.Labels = resourceQuota.Labels
	editResourceQuota.ResourceVersion = resourceQuota.ResourceVersion
	editResourceQuota.Annotations = resourceQuota.Annotations
	editResourceQuota.Spec = resourceQuota.Spec

	if errs := validation.ValidateResourceQuota(editResourceQuota); len(errs) > 0 {
		return nil, false, errors.NewInvalid("resourceQuota", editResourceQuota.Name, errs)
	}

	if err := rs.registry.UpdateWithName(ctx, editResourceQuota.Name, editResourceQuota); err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, editResourceQuota.Name)
	return out, false, err
}

// Delete deletes the ResourceQuota with the specified name
func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.ResourceQuota)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return rs.registry.Delete(ctx, name)
}

// Get gets a ResourceQuota with the specified name
func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	resourceQuota, ok := obj.(*api.ResourceQuota)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return resourceQuota, err
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	return labels.Set{}, labels.Set{}, nil
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return rs.registry.List(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.Watch(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}

// New returns a new api.ResourceQuota
func (*REST) New() runtime.Object {
	return &api.ResourceQuota{}
}

func (*REST) NewList() runtime.Object {
	return &api.ResourceQuotaList{}
}
