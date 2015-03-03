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

package persistentvolume

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST implements the RESTStorage interface in terms of a PodRegistry.
type REST struct {
	registry generic.Registry
}

type RESTConfig struct {
	Registry generic.Registry
}

// NewREST returns a new REST.
func NewREST(registry generic.Registry) *REST {
	return &REST{
		registry: registry,
	}
}

func (*REST) New() runtime.Object {
	return &api.PersistentVolume{}
}

func (*REST) NewList() runtime.Object {
	return &api.PersistentVolumeList{}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	persistentvolume := obj.(*api.PersistentVolume)
	api.FillObjectMetaSystemFields(ctx, &persistentvolume.ObjectMeta)
	if errs := validation.ValidatePersistentVolume(persistentvolume); len(errs) > 0 {
		return nil, errors.NewInvalid("persistentVolume", persistentvolume.Name, errs)
	}

	err := rs.registry.CreateWithName(ctx, persistentvolume.Name, persistentvolume)
	if err != nil {
		return nil, err
	}

	return rs.registry.Get(ctx, persistentvolume.Name)
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	persistentvolume, ok := obj.(*api.PersistentVolume)
	if !ok {
		return nil, false, fmt.Errorf("invalid object type")
	}
	if errs := validation.ValidatePersistentVolume(persistentvolume); len(errs) > 0 {
		return nil, false, errors.NewInvalid("persistentVolume", persistentvolume.Name, errs)
	}

	err := rs.registry.UpdateWithName(ctx, persistentvolume.Name, persistentvolume)
	if err != nil {
		return nil, false, err
	}

	out, err := rs.registry.Get(ctx, persistentvolume.Name)
	return out, false, err
}

func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	persistentvolume, ok := obj.(*api.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("Invalid object type")
	}
	return persistentvolume, err
}

func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	return rs.registry.Delete(ctx, name)
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
