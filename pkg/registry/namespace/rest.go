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

package namespace

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kerrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST provides the RESTStorage access patterns to work with Namespace objects.
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

// Create creates a Namespace object
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	namespace := obj.(*api.Namespace)
	if err := rest.BeforeCreate(rest.Namespaces, ctx, obj); err != nil {
		return nil, err
	}
	if err := rs.registry.CreateWithName(ctx, namespace.Name, namespace); err != nil {
		err = rest.CheckGeneratedNameError(rest.Namespaces, err, namespace)
		return nil, err
	}
	return rs.registry.Get(ctx, namespace.Name)
}

// Update updates a Namespace object.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	namespace, ok := obj.(*api.Namespace)
	if !ok {
		return nil, false, fmt.Errorf("not a namespace: %#v", obj)
	}

	oldObj, err := rs.registry.Get(ctx, namespace.Name)
	if err != nil {
		return nil, false, err
	}

	oldNamespace := oldObj.(*api.Namespace)
	if errs := validation.ValidateNamespaceUpdate(oldNamespace, namespace); len(errs) > 0 {
		return nil, false, kerrors.NewInvalid("namespace", namespace.Name, errs)
	}

	if err := rs.registry.UpdateWithName(ctx, oldNamespace.Name, oldNamespace); err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, oldNamespace.Name)
	return out, false, err
}

// Delete deletes the Namespace with the specified name
func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.Namespace)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return rs.registry.Delete(ctx, name)
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	namespace, ok := obj.(*api.Namespace)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return namespace, err
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

// New returns a new api.Namespace
func (*REST) New() runtime.Object {
	return &api.Namespace{}
}

func (*REST) NewList() runtime.Object {
	return &api.NamespaceList{}
}
