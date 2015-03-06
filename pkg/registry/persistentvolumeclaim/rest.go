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

package persistentvolumeclaim

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
	return &api.PersistentVolumeClaim{}
}

func (*REST) NewList() runtime.Object {
	return &api.PersistentVolumeClaimList{}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	persistentvolumeclaim, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &persistentvolumeclaim.ObjectMeta) {
		return nil, errors.NewConflict("persistentVolumeClaim", persistentvolumeclaim.Namespace, fmt.Errorf("PersistentVolumeClaim.Namespace does not match the provided context"))
	}

	api.FillObjectMetaSystemFields(ctx, &persistentvolumeclaim.ObjectMeta)
	if errs := validation.ValidatePersistentVolumeClaim(persistentvolumeclaim); len(errs) > 0 {
		return nil, errors.NewInvalid("persistentVolumeClaim", persistentvolumeclaim.Name, errs)
	}
	err := rs.registry.CreateWithName(ctx, persistentvolumeclaim.Name, persistentvolumeclaim)
	if err != nil {
		return nil, err
	}

	return rs.registry.Get(ctx, persistentvolumeclaim.Name)
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	persistentvolumeclaim, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, false, fmt.Errorf("invalid object type")
	}
	if !api.ValidNamespace(ctx, &persistentvolumeclaim.ObjectMeta) {
		return nil, false, errors.NewConflict("persistentVolumeClaim", persistentvolumeclaim.Namespace, fmt.Errorf("PersistentStorageController.Namespace does not match the provided context"))
	}
	if errs := validation.ValidatePersistentVolumeClaim(persistentvolumeclaim); len(errs) > 0 {
		return nil, false, errors.NewInvalid("persistentVolumeClaim", persistentvolumeclaim.Name, errs)
	}
	if err := rs.registry.UpdateWithName(ctx, persistentvolumeclaim.Name, persistentvolumeclaim); err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, persistentvolumeclaim.Name)
	return out, false, err
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	persistentvolumeclaim, err := rs.registry.Get(ctx, id)
	if err != nil {
		return persistentvolumeclaim, err
	}
	_, ok := persistentvolumeclaim.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return persistentvolumeclaim, err
}

func (rs *REST) Delete(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}

	_, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return rs.registry.Delete(ctx, id)
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	return labels.Set{}, labels.Set{}, nil
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return rs.registry.ListPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}
