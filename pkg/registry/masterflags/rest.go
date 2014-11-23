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

package masterflags

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST adapts a masterFlags registry into apiserver's RESTStorage model.
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

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	masterFlags, ok := obj.(*api.MasterFlags)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	if api.Namespace(ctx) != "" {
		if !api.ValidNamespace(ctx, &masterFlags.ObjectMeta) {
			return nil, errors.NewConflict("masterFlags", masterFlags.Namespace, fmt.Errorf("masterFlags.namespace does not match the provided context"))
		}
	}
	if errs := validation.ValidateMasterFlags(masterFlags); len(errs) > 0 {
		return nil, errors.NewInvalid("masterFlags", masterFlags.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &masterFlags.ObjectMeta)

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		err := rs.registry.Create(ctx, masterFlags.Name, masterFlags)
		if err != nil {
			return nil, err
		}
		return rs.registry.Get(ctx, masterFlags.Name)
	}), nil
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.MasterFlags)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.Delete(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	masterFlags, ok := obj.(*api.MasterFlags)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return masterFlags, err
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	_, ok := obj.(*api.MasterFlags)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type")
	}
	return labels.Set{}, labels.Set{}, nil
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return rs.registry.List(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

// Watch returns MasterFlagss masterFlagss via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.Watch(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}

// New returns a new api.MasterFlags
func (*REST) New() runtime.Object {
	return &api.MasterFlags{}
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	masterFlags, ok := obj.(*api.MasterFlags)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	if api.Namespace(ctx) != "" {
		if !api.ValidNamespace(ctx, &masterFlags.ObjectMeta) {
			return nil, errors.NewConflict("masterFlags", masterFlags.Namespace, fmt.Errorf("masterFlags.namespace does not match the provided context"))
		}
	}
	if errs := validation.ValidateMasterFlags(masterFlags); len(errs) > 0 {
		return nil, errors.NewInvalid("masterFlags", masterFlags.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &masterFlags.ObjectMeta)

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		err := rs.registry.Update(ctx, masterFlags.Name, masterFlags)
		if err != nil {
			return nil, err
		}
		return rs.registry.Get(ctx, masterFlags.Name)
	}), nil
}
