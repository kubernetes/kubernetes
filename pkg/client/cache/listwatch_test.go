/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cache

import (
	"net/http/httptest"
	"net/url"
	"path"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func parseSelectorOrDie(s string) fields.Selector {
	selector, err := fields.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// buildResourcePath is a convenience function for knowing if a namespace should be in a path param or not
func buildResourcePath(prefix, namespace, resource string) string {
	base := path.Join("/api", testapi.Version(), prefix)
	if len(namespace) > 0 {
		if !(testapi.Version() == "v1beta1" || testapi.Version() == "v1beta2") {
			base = path.Join(base, "namespaces", namespace)
		}
	}
	return path.Join(base, resource)
}

func getHostFieldLabel() string {
	switch testapi.Version() {
	case "v1beta1", "v1beta2":
		return "DesiredState.Host"
	default:
		return "spec.host"
	}
}

// buildQueryValues is a convenience function for knowing if a namespace should be in a query param or not
func buildQueryValues(namespace string, query url.Values) url.Values {
	v := url.Values{}
	if query != nil {
		for key, values := range query {
			for _, value := range values {
				v.Add(key, value)
			}
		}
	}
	if testapi.Version() == "v1beta1" || testapi.Version() == "v1beta2" {
		v.Set("namespace", namespace)
	}
	return v
}

func buildLocation(resourcePath string, query url.Values) string {
	return resourcePath + "?" + query.Encode()
}

func TestListWatchesCanList(t *testing.T) {
	fieldSelectorQueryParamName := api.FieldSelectorQueryParam(testapi.Version())
	table := []struct {
		location      string
		resource      string
		namespace     string
		fieldSelector fields.Selector
	}{
		// Minion
		{
			location:      buildLocation(buildResourcePath("", api.NamespaceAll, "minions"), buildQueryValues(api.NamespaceAll, nil)),
			resource:      "minions",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				buildResourcePath("", api.NamespaceAll, "pods"),
				buildQueryValues(api.NamespaceAll, url.Values{fieldSelectorQueryParamName: []string{getHostFieldLabel() + "="}})),
			resource:      "pods",
			namespace:     api.NamespaceAll,
			fieldSelector: fields.Set{getHostFieldLabel(): ""}.AsSelector(),
		},
		// pod in namespace "foo"
		{
			location: buildLocation(
				buildResourcePath("", "foo", "pods"),
				buildQueryValues("foo", url.Values{fieldSelectorQueryParamName: []string{getHostFieldLabel() + "="}})),
			resource:      "pods",
			namespace:     "foo",
			fieldSelector: fields.Set{getHostFieldLabel(): ""}.AsSelector(),
		},
	}
	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Version()})
		lw := NewListWatchFromClient(client, item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.List()
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

func TestListWatchesCanWatch(t *testing.T) {
	fieldSelectorQueryParamName := api.FieldSelectorQueryParam(testapi.Version())
	table := []struct {
		rv            string
		location      string
		resource      string
		namespace     string
		fieldSelector fields.Selector
	}{
		// Minion
		{
			location: buildLocation(
				buildResourcePath("watch", api.NamespaceAll, "minions"),
				buildQueryValues(api.NamespaceAll, url.Values{"resourceVersion": []string{""}})),
			rv:            "",
			resource:      "minions",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		{
			location: buildLocation(
				buildResourcePath("watch", api.NamespaceAll, "minions"),
				buildQueryValues(api.NamespaceAll, url.Values{"resourceVersion": []string{"42"}})),
			rv:            "42",
			resource:      "minions",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				buildResourcePath("watch", api.NamespaceAll, "pods"),
				buildQueryValues(api.NamespaceAll, url.Values{fieldSelectorQueryParamName: []string{getHostFieldLabel() + "="}, "resourceVersion": []string{"0"}})),
			rv:            "0",
			resource:      "pods",
			namespace:     api.NamespaceAll,
			fieldSelector: fields.Set{getHostFieldLabel(): ""}.AsSelector(),
		},
		// pod with namespace foo and assigned field selector
		{
			location: buildLocation(
				buildResourcePath("watch", "foo", "pods"),
				buildQueryValues("foo", url.Values{fieldSelectorQueryParamName: []string{getHostFieldLabel() + "="}, "resourceVersion": []string{"0"}})),
			rv:            "0",
			resource:      "pods",
			namespace:     "foo",
			fieldSelector: fields.Set{getHostFieldLabel(): ""}.AsSelector(),
		},
	}

	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Version()})
		lw := NewListWatchFromClient(client, item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.Watch(item.rv)
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}
