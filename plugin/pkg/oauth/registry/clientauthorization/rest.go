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

package clientauthorization

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api/validation"
)

// REST implements the RESTStorage interface in terms of an Registry.
type REST struct {
	registry Registry
}

// NewStorage returns a new REST.
func NewREST(registry Registry) apiserver.RESTStorage {
	return &REST{registry}
}

// New returns a new ClientAuthorization for use with Create and Update.
func (s *REST) New() runtime.Object {
	return &oapi.OAuthClientAuthorization{}
}

// Create registers the given ClientAuthorization.
func (s *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	authorization, ok := obj.(*oapi.OAuthClientAuthorization)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}

	authorization.Name = s.registry.Name(authorization.UserName, authorization.ClientName)
	if errs := validation.ValidateClientAuthorization(authorization); len(errs) > 0 {
		return nil, errors.NewInvalid("clientAuthorization", authorization.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &authorization.ObjectMeta)

	if err := s.registry.CreateWithName(ctx, authorization.Name, authorization); err != nil {
		return nil, err
	}
	return s.registry.Get(ctx, authorization.Name)
}

// Get retrieves an ClientAuthorization by id.
func (s *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	return s.registry.Get(ctx, id)
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	auth, ok := obj.(*oapi.OAuthClientAuthorization)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type")
	}
	return labels.Set(auth.Labels), labels.Set{
		"clientName": auth.ClientName,
		"userName":   auth.UserName,
		"userUID":    auth.UserUID,
	}, nil
}

// List retrieves a list of ClientAuthorizations that match selector.
func (s *REST) List(ctx api.Context, labels, fields labels.Selector) (runtime.Object, error) {
	return s.registry.List(ctx, &generic.SelectionPredicate{labels, fields, s.getAttrs})
}

// Update modifies an existing client authorization
func (s *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	authorization, ok := obj.(*oapi.OAuthClientAuthorization)
	if !ok {
		return nil, fmt.Errorf("not an authorization: %#v", obj)
	}

	if errs := validation.ValidateClientAuthorization(authorization); len(errs) > 0 {
		return nil, errors.NewInvalid("clientAuthorization", authorization.Name, errs)
	}

	oldobj, err := s.registry.Get(ctx, authorization.Name)
	if err != nil {
		return nil, err
	}
	oldauth, ok := oldobj.(*oapi.OAuthClientAuthorization)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	if errs := validation.ValidateClientAuthorizationUpdate(authorization, oldauth); len(errs) > 0 {
		return nil, errors.NewInvalid("clientAuthorization", authorization.Name, errs)
	}

	if err := s.registry.UpdateWithName(ctx, authorization.Name, authorization); err != nil {
		return nil, err
	}
	return s.registry.Get(ctx, authorization.Name)
}

// Delete asynchronously deletes an ClientAuthorization specified by its id.
func (s *REST) Delete(ctx api.Context, id string) (runtime.Object, error) {
	obj, err := s.registry.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*oapi.OAuthClientAuthorization)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return s.registry.Delete(ctx, id)
}
