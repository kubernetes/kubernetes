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

package secret

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

// REST provides the RESTStorage access patterns to work with Secret objects.
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

// Create a Secret object
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	secret, ok := obj.(*api.Secret)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	if !api.ValidNamespace(ctx, &secret.ObjectMeta) {
		return nil, errors.NewConflict("secret", secret.Namespace, fmt.Errorf("Secret.Namespace does not match the provided context"))
	}

	if len(secret.Name) == 0 {
		secret.Name = string(util.NewUUID())
	}

	if errs := validation.ValidateSecret(secret); len(errs) > 0 {
		return nil, errors.NewInvalid("secret", secret.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &secret.ObjectMeta)

	err := rs.registry.CreateWithName(ctx, secret.Name, secret)
	if err != nil {
		return nil, err
	}
	return rs.registry.Get(ctx, secret.Name)
}

// Update updates a Secret object.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	secret, ok := obj.(*api.Secret)
	if !ok {
		return nil, fmt.Errorf("not a secret: %#v", obj)
	}

	if !api.ValidNamespace(ctx, &secret.ObjectMeta) {
		return nil, errors.NewConflict("secret", secret.Namespace, fmt.Errorf("Secret.Namespace does not match the provided context"))
	}

	oldObj, err := rs.registry.Get(ctx, secret.Name)
	if err != nil {
		return nil, err
	}

	editSecret := oldObj.(*api.Secret)

	// set the editable fields on the existing object
	editSecret.Labels = secret.Labels
	editSecret.ResourceVersion = secret.ResourceVersion
	editSecret.Annotations = secret.Annotations

	if errs := validation.ValidateSecret(editSecret); len(errs) > 0 {
		return nil, errors.NewInvalid("secret", editSecret.Name, errs)
	}

	err = rs.registry.UpdateWithName(ctx, editSecret.Name, editSecret)
	if err != nil {
		return nil, err
	}
	return rs.registry.Get(ctx, editSecret.Name)
}

// Delete deletes the Secret with the specified name
func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.Secret)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	return rs.registry.Delete(ctx, name)
}

// Get gets a Secret with the specified name
func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	secret, ok := obj.(*api.Secret)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return secret, err
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

// New returns a new api.Secret
func (*REST) New() runtime.Object {
	return &api.Secret{}
}

func (*REST) NewList() runtime.Object {
	return &api.SecretList{}
}
