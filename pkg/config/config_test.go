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

package config

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func getTyperAndMapper() (runtime.ObjectTyper, meta.RESTMapper) {
	return api.Scheme, latest.RESTMapper
}

func getFakeClient(t *testing.T, validURLs []string) (ClientFunc, *httptest.Server) {
	handlerFunc := func(w http.ResponseWriter, r *http.Request) {
		for _, u := range validURLs {
			if u == r.RequestURI {
				return
			}
		}
		t.Errorf("Unexpected HTTP request: %s, expected %v", r.RequestURI, validURLs)
	}
	server := httptest.NewServer(http.HandlerFunc(handlerFunc))
	return func(mapping *meta.RESTMapping) (*client.RESTClient, error) {
		fakeCodec := runtime.CodecFor(api.Scheme, "v1beta1")
		fakeUri, _ := url.Parse(server.URL + "/api/v1beta1")
		return client.NewRESTClient(fakeUri, fakeCodec), nil
	}, server
}

func TestCreateObjects(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Pod{
		TypeMeta:   api.TypeMeta{APIVersion: "v1beta1", Kind: "Pod"},
		ObjectMeta: api.ObjectMeta{Name: "test-pod"},
	})

	items = append(items, &api.Service{
		TypeMeta:   api.TypeMeta{APIVersion: "v1beta1", Kind: "Service"},
		ObjectMeta: api.ObjectMeta{Name: "test-service"},
	})

	typer, mapper := getTyperAndMapper()
	client, s := getFakeClient(t, []string{"/api/v1beta1/pods", "/api/v1beta1/services"})

	errs := CreateObjects(typer, mapper, client, items)
	s.Close()
	if len(errs) != 0 {
		t.Errorf("Unexpected errors during config.Create(): %v", errs)
	}
}

func TestCreateNoNameItem(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Service{
		TypeMeta: api.TypeMeta{APIVersion: "v1beta1", Kind: "Service"},
	})

	typer, mapper := getTyperAndMapper()
	client, s := getFakeClient(t, []string{"/api/v1beta1/services"})

	errs := CreateObjects(typer, mapper, client, items)
	s.Close()

	if len(errs) == 0 {
		t.Errorf("Expected required value error for missing name")
	}

	e := errs[0].(errors.ValidationError)
	if errors.ValueOf(e.Type) != "required value" {
		t.Errorf("Expected ValidationErrorTypeRequired error, got %#v", e)
	}

	if e.Field != "Config.item[0].name" {
		t.Errorf("Expected 'Config.item[0].name' as error field, got '%#v'", e.Field)
	}
}

type InvalidItem struct{}

func (*InvalidItem) IsAnAPIObject() {}

func TestCreateInvalidItem(t *testing.T) {
	items := []runtime.Object{
		&InvalidItem{},
	}

	typer, mapper := getTyperAndMapper()
	client, s := getFakeClient(t, []string{})

	errs := CreateObjects(typer, mapper, client, items)
	s.Close()

	if len(errs) == 0 {
		t.Errorf("Expected invalid value error for kind")
	}

	e := errs[0].(errors.ValidationError)
	if errors.ValueOf(e.Type) != "invalid value" {
		t.Errorf("Expected ValidationErrorTypeInvalid error, got %#v", e)
	}

	if e.Field != "Config.item[0].kind" {
		t.Errorf("Expected 'Config.item[0].kind' as error field, got '%#v'", e.Field)
	}
}

func TestCreateNoClientItems(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Pod{
		TypeMeta:   api.TypeMeta{APIVersion: "v1beta1", Kind: "Pod"},
		ObjectMeta: api.ObjectMeta{Name: "test-pod"},
	})

	typer, mapper := getTyperAndMapper()
	_, s := getFakeClient(t, []string{"/api/v1beta1/pods", "/api/v1beta1/services"})

	noClientFunc := func(mapping *meta.RESTMapping) (*client.RESTClient, error) {
		return nil, fmt.Errorf("no client")
	}

	errs := CreateObjects(typer, mapper, noClientFunc, items)
	s.Close()

	if len(errs) == 0 {
		t.Errorf("Expected invalid value error for client")
	}

	e := errs[0].(errors.ValidationError)
	if errors.ValueOf(e.Type) != "unsupported value" {
		t.Errorf("Expected ValidationErrorTypeUnsupported error, got %#v", e)
	}

	if e.Field != "Config.item[0].client" {
		t.Errorf("Expected 'Config.item[0].client' as error field, got '%#v'", e.Field)
	}
}
