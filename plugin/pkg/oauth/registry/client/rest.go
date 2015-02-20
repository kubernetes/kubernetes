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

package client

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

// New returns a new Client for use with Create and Update.
func (s *REST) New() runtime.Object {
	return &oapi.OAuthClient{}
}

// Get retrieves an Client by name.
func (s *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	client, err := s.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return client, nil
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	client, ok := obj.(*oapi.OAuthClient)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type")
	}
	return labels.Set(client.Labels), labels.Set{
		"name": client.Name,
	}, nil
}

// List retrieves a list of Clients that match selector.
func (s *REST) List(ctx api.Context, labels, fields labels.Selector) (runtime.Object, error) {
	return s.registry.List(ctx, &generic.SelectionPredicate{labels, fields, s.getAttrs})
}

// Create registers the given Client.
func (s *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	client, ok := obj.(*oapi.OAuthClient)
	if !ok {
		return nil, fmt.Errorf("not an client: %#v", obj)
	}

	if errs := validation.ValidateClient(client); len(errs) > 0 {
		return nil, errors.NewInvalid("client", client.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &client.ObjectMeta)

	if err := s.registry.CreateWithName(ctx, client.Name, client); err != nil {
		return nil, err
	}
	return s.Get(ctx, client.Name)
}

// Update is not supported for Clients, as they are immutable.
func (s *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	return nil, fmt.Errorf("Clients may not be changed.")
}

// Delete asynchronously deletes an Client specified by its name.
func (s *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	return s.registry.Delete(ctx, name)
}
