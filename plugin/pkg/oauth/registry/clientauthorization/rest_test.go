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
	"errors"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/test"
)

func TestCreateValidationError(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	storage := REST{
		registry: &registry,
	}
	clientAuth := &oapi.OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
		// ClientName: "clientName",// Missing required field
		UserName: "userName",
		UserUID:  "userUID",
	}

	ctx := api.NewContext()
	_, err := storage.Create(ctx, clientAuth)
	if err == nil {
		t.Errorf("Expected validation error")
	}
}

func TestCreateStorageError(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	registry.Err = errors.New("Sample Error")

	storage := REST{
		registry: &registry,
	}
	clientAuth := &oapi.OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{Name: "clientName"},
		ClientName: "clientName",
		UserName:   "userName",
		UserUID:    "userUID",
	}

	ctx := api.NewContext()
	_, err := storage.Create(ctx, clientAuth)
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
	if err != registry.Err {
		t.Fatalf("Expected error %v, got %v", registry.Err, err)
	}
}

func TestCreateValid(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	storage := REST{
		registry: &registry,
	}
	clientAuth := &oapi.OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{Name: "clientName"},
		ClientName: "clientName",
		UserName:   "userName",
		UserUID:    "userUID",
	}

	ctx := api.NewContext()
	r, err := storage.Create(ctx, clientAuth)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	switch r := r.(type) {
	case *api.Status:
		t.Errorf("Got back unexpected status: %#v", r)
	case *oapi.OAuthClientAuthorization:
		// expected case
	default:
		t.Errorf("Got unexpected type: %#v", r)
	}
}

func TestGetError(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
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
	registry := test.NewClientAuthorizationRegistry()
	registry.Object = &oapi.OAuthClientAuthorization{ObjectMeta: api.ObjectMeta{Name: "clientName"}}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	clientAuth, err := storage.Get(ctx, "name")
	if err != nil {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	if clientAuth != registry.Object {
		t.Errorf("got unexpected clientAuthorization: %v", clientAuth)
		return
	}
}

func TestListError(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
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
	registry := test.NewClientAuthorizationRegistry()
	registry.ObjectList = &oapi.OAuthClientAuthorizationList{}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	clientAuths, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	switch clientAuths := clientAuths.(type) {
	case *oapi.OAuthClientAuthorizationList:
		if len(clientAuths.Items) != 0 {
			t.Errorf("expected empty list, got %#v", clientAuths)
		}
	default:
		t.Errorf("expected clientAuthList, got: %v", clientAuths)
		return
	}
}

func TestList(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	registry.ObjectList = &oapi.OAuthClientAuthorizationList{
		Items: []oapi.OAuthClientAuthorization{
			{},
			{},
		},
	}
	storage := REST{
		registry: &registry,
	}
	ctx := api.NewContext()
	clientAuths, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != registry.Err {
		t.Errorf("got unexpected error: %v", err)
		return
	}
	switch clientAuths := clientAuths.(type) {
	case *oapi.OAuthClientAuthorizationList:
		if len(clientAuths.Items) != 2 {
			t.Errorf("expected list with 2 items, got %#v", clientAuths)
		}
	default:
		t.Errorf("expected clientAuthList, got: %v", clientAuths)
		return
	}
}

func TestUpdateNotSupported(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	registry.Err = errors.New("Storage Error")
	storage := REST{
		registry: &registry,
	}
	client := &oapi.OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{Name: "clientName"},
	}

	ctx := api.NewContext()
	_, err := storage.Update(ctx, client)
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
	registry := test.NewClientAuthorizationRegistry()
	registry.Err = errors.New("Sample Error")
	storage := REST{
		registry: &registry,
	}

	ctx := api.NewContext()
	_, err := storage.Delete(ctx, "foo")
	if err == nil {
		t.Errorf("unexpected success")
		return
	}
}

func TestDeleteValid(t *testing.T) {
	registry := test.NewClientAuthorizationRegistry()
	registry.Object = &oapi.OAuthClientAuthorization{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	storage := REST{
		registry: &registry,
	}

	ctx := api.NewContext()
	r, err := storage.Delete(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	switch r := r.(type) {
	case *api.Status:
		if r.Status != "Success" {
			t.Fatalf("Got back non-success status: %#v", r)
		}
	default:
		t.Fatalf("Got back non-status result: %v", r)
	}
}
