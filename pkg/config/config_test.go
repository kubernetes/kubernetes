/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func getTyperAndMapper() (runtime.ObjectTyper, meta.RESTMapper) {
	return api.Scheme, latest.RESTMapper
}

func getFakeClient(t *testing.T, validURLs []string) (ClientPosterFunc, *httptest.Server) {
	handlerFunc := func(w http.ResponseWriter, r *http.Request) {
		for _, u := range validURLs {
			if u == r.RequestURI {
				return
			}
		}
		t.Errorf("Unexpected HTTP request: %s, expected %v", r.RequestURI, validURLs)
	}
	server := httptest.NewServer(http.HandlerFunc(handlerFunc))
	return func(mapping *meta.RESTMapping) (RESTClientPoster, error) {
		fakeCodec := testapi.Codec()
		fakeUri, _ := url.Parse(server.URL + "/api/" + testapi.Version())
		legacyBehavior := api.PreV1Beta3(testapi.Version())
		return client.NewRESTClient(fakeUri, testapi.Version(), fakeCodec, legacyBehavior, 5, 10), nil
	}, server
}

func TestCreateObjects(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "test-pod", Namespace: "default"},
	})

	items = append(items, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "test-service", Namespace: "default"},
	})

	typer, mapper := getTyperAndMapper()
	client, s := getFakeClient(t, []string{
		testapi.ResourcePathWithNamespaceQuery("pods", api.NamespaceDefault, ""),
		testapi.ResourcePathWithNamespaceQuery("services", api.NamespaceDefault, ""),
	})

	errs := CreateObjects(typer, mapper, client, items)
	s.Close()
	if len(errs) != 0 {
		t.Errorf("Unexpected errors during config.Create(): %v", errs)
	}
}

func TestCreateNoNameItem(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Service{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Version(), Kind: "Service"},
	})

	typer, mapper := getTyperAndMapper()
	client, s := getFakeClient(t, []string{
		testapi.ResourcePath("services", api.NamespaceDefault, ""),
	})

	errs := CreateObjects(typer, mapper, client, items)
	s.Close()

	if len(errs) == 0 {
		t.Errorf("Expected required value error for missing name")
	}

	errStr := errs[0].Error()
	if !strings.Contains(errStr, "Config.item[0]: name") {
		t.Errorf("Expected 'Config.item[0]: name' in error string, got '%s'", errStr)
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

	errStr := errs[0].Error()
	if !strings.Contains(errStr, "Config.item[0] kind") {
		t.Errorf("Expected 'Config.item[0] kind' in error string, got '%s'", errStr)
	}
}

func TestCreateNoClientItems(t *testing.T) {
	items := []runtime.Object{}

	items = append(items, &api.Pod{
		TypeMeta:   api.TypeMeta{APIVersion: testapi.Version(), Kind: "Pod"},
		ObjectMeta: api.ObjectMeta{Name: "test-pod"},
	})

	typer, mapper := getTyperAndMapper()
	_, s := getFakeClient(t, []string{
		testapi.ResourcePath("pods", api.NamespaceDefault, ""),
		testapi.ResourcePath("services", api.NamespaceDefault, ""),
	})

	noClientFunc := func(mapping *meta.RESTMapping) (RESTClientPoster, error) {
		return nil, fmt.Errorf("no client")
	}

	errs := CreateObjects(typer, mapper, noClientFunc, items)
	s.Close()

	if len(errs) == 0 {
		t.Errorf("Expected invalid value error for client")
	}

	errStr := errs[0].Error()
	if !strings.Contains(errStr, "Config.item[0] client") {
		t.Errorf("Expected 'Config.item[0] client' in error string, got '%s'", errStr)
	}
}
