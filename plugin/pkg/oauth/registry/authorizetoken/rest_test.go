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

package authorizetoken

import (
	"errors"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/test"
)

func TestCreateValidationError(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	storage := REST{
		registry: &registry,
	}
	authorizeToken := &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
		// ClientName: "clientName", // Missing required field
		UserName: "userName",
		UserUID:  "userUID",
	}

	ctx := api.NewContext()
	_, err := storage.Create(ctx, authorizeToken)
	if err == nil {
		t.Errorf("Expected validation error")
	}
}

func TestCreateStorageError(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Err = errors.New("Sample Error")
	storage := REST{
		registry: &registry,
	}
	authorizeToken := &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
		ClientName: "clientName",
		UserName:   "userName",
		UserUID:    "userUID",
	}

	ctx := api.NewContext()
	_, err := storage.Create(ctx, authorizeToken)
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
	if err != registry.Err {
		t.Fatalf("Expected error %v, got %v", registry.Err, err)
	}
}

func TestCreateValid(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	storage := REST{
		registry: &registry,
	}
	authorizeToken := &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
		ClientName: "clientName",
		UserName:   "userName",
		UserUID:    "userUID",
	}

	ctx := api.NewContext()
	r, err := storage.Create(ctx, authorizeToken)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	switch r := r.(type) {
	case *api.Status:
		t.Errorf("Got back unexpected status: %#v", r)
	case *oapi.OAuthAuthorizeToken:
		// expected case
	default:
		t.Errorf("Got unexpected type: %#v", r)
	}
}

func TestGetError(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Err = errors.New("Sample Error")
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	_, err := storage.Get(ctx, "name")
	if err == nil {
		t.Errorf("expected error")
		return
	}
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
}

func TestGetValid(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Object = &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "tokenName"},
	}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	token, err := storage.Get(ctx, "name")
	if err != nil {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	if token != registry.Object {
		t.Errorf("got unexpected token: %v", token)
		return
	}
}

func TestListError(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Err = errors.New("Sample Error")
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	_, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err == nil {
		t.Errorf("expected error")
		return
	}
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
}

func TestListEmpty(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.ObjectList = &oapi.OAuthAuthorizeTokenList{}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	tokens, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	switch tokens := tokens.(type) {
	case *oapi.OAuthAuthorizeTokenList:
		if len(tokens.Items) != 0 {
			t.Errorf("expected empty list, got %#v", tokens)
		}
	default:
		t.Errorf("expected AuthorizeTokenList, got: %v", tokens)
		return
	}
}

func TestList(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.ObjectList = &oapi.OAuthAuthorizeTokenList{
		Items: []oapi.OAuthAuthorizeToken{
			{},
			{},
		},
	}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	tokens, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	switch tokens := tokens.(type) {
	case *oapi.OAuthAuthorizeTokenList:
		if len(tokens.Items) != 2 {
			t.Errorf("expected list with 2 items, got %#v", tokens)
		}
	default:
		t.Errorf("expected AuthorizeTokenList, got: %v", tokens)
		return
	}
}

func TestUpdateNotSupported(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Err = errors.New("Storage Error")
	storage := REST{
		registry: &registry,
	}
	authorizeToken := &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
		ClientName: "clientName",
		UserName:   "userName",
		UserUID:    "userUID",
	}

	ctx := api.NewContext()
	_, err := storage.Update(ctx, authorizeToken)
	if err == nil {
		t.Errorf("expected unsupported error, but update succeeded")
		return
	}
	if err == registry.Err {
		t.Errorf("expected unsupported error, but registry was called")
		return
	}
}

func TestDeleteError(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	registry.Err = errors.New("Sample Error")
	storage := REST{
		registry: &registry,
	}

	ctx := api.NewContext()
	_, err := storage.Delete(ctx, "foo")
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
	if err != registry.Err {
		t.Fatalf("Expected error %v, got %v", registry.Err, err)
	}
}

func TestDeleteValid(t *testing.T) {
	registry := test.NewAuthorizeTokenRegistry()
	storage := REST{
		registry: &registry,
	}

	ctx := api.NewContext()
	r, err := storage.Delete(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	switch r := r.(type) {
	case *api.Status:
		if r.Status != "Success" {
			t.Errorf("Got back non-success status: %#v", r)
		}
	default:
		t.Errorf("Got back non-status result: %v", r)
	}
}
