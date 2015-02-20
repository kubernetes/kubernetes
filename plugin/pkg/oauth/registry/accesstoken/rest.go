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

package accesstoken

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

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

// New returns a new AccessToken for use with Create and Update.
func (s *REST) New() runtime.Object {
	return &oapi.OAuthAccessToken{}
}

// Get retrieves an AccessToken by name.
func (s *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	token, err := s.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return token, nil
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels, objFields labels.Set, err error) {
	token, ok := obj.(*oapi.OAuthAccessToken)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type")
	}
	return labels.Set(token.Labels), labels.Set{
		"name":           token.Name,
		"clientName":     token.ClientName,
		"userName":       token.UserName,
		"userUID":        token.UserUID,
		"authorizeToken": token.AuthorizeToken.Name,
		"refreshToken":   token.RefreshToken,
	}, nil
}

// List retrieves a list of AccessTokens that match selector.
func (s *REST) List(ctx api.Context, labels, fields labels.Selector) (runtime.Object, error) {
	return s.registry.List(ctx, &generic.SelectionPredicate{labels, fields, s.getAttrs})
}

// Create registers the given AccessToken.
func (s *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	token, ok := obj.(*oapi.OAuthAccessToken)
	if !ok {
		return nil, fmt.Errorf("not an token: %#v", obj)
	}

	if errs := validation.ValidateAccessToken(token); len(errs) > 0 {
		return nil, errors.NewInvalid("accessToken", token.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &token.ObjectMeta)

	token.CreationTimestamp = util.Now()

	if err := s.registry.CreateWithName(ctx, token.Name, token); err != nil {
		return nil, err
	}
	return s.Get(ctx, token.Name)
}

// Update is not supported for AccessTokens, as they are immutable.
func (s *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	return nil, fmt.Errorf("AccessTokens may not be changed.")
}

// Delete asynchronously deletes an AccessToken specified by its name.
func (s *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	return s.registry.Delete(ctx, name)
}
