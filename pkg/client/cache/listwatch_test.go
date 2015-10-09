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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util"
)

func parseSelectorOrDie(s string) fields.Selector {
	selector, err := fields.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// buildQueryValues is a convenience function for knowing if a namespace should be in a query param or not
func buildQueryValues(query url.Values) url.Values {
	v := url.Values{}
	if query != nil {
		for key, values := range query {
			for _, value := range values {
				v.Add(key, value)
			}
		}
	}
	return v
}

func buildLocation(resourcePath string, query url.Values) string {
	return resourcePath + "?" + query.Encode()
}

func TestListWatchesCanList(t *testing.T) {
	fieldSelectorQueryParamName := unversioned.FieldSelectorQueryParam(testapi.Default.Version())
	table := []struct {
		location      string
		resource      string
		namespace     string
		fieldSelector fields.Selector
	}{
		// Node
		{
			location:      testapi.Default.ResourcePath("nodes", api.NamespaceAll, ""),
			resource:      "nodes",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				testapi.Default.ResourcePath("pods", api.NamespaceAll, ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}})),
			resource:      "pods",
			namespace:     api.NamespaceAll,
			fieldSelector: fields.Set{"spec.host": ""}.AsSelector(),
		},
		// pod in namespace "foo"
		{
			location: buildLocation(
				testapi.Default.ResourcePath("pods", "foo", ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}})),
			resource:      "pods",
			namespace:     "foo",
			fieldSelector: fields.Set{"spec.host": ""}.AsSelector(),
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
		client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Default.Version()})
		lw := NewListWatchFromClient(client, item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.List()
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

func TestListWatchesCanWatch(t *testing.T) {
	fieldSelectorQueryParamName := unversioned.FieldSelectorQueryParam(testapi.Default.Version())
	table := []struct {
		rv            string
		location      string
		resource      string
		namespace     string
		fieldSelector fields.Selector
	}{
		// Node
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "nodes", api.NamespaceAll, ""),
				buildQueryValues(url.Values{"resourceVersion": []string{""}})),
			rv:            "",
			resource:      "nodes",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "nodes", api.NamespaceAll, ""),
				buildQueryValues(url.Values{"resourceVersion": []string{"42"}})),
			rv:            "42",
			resource:      "nodes",
			namespace:     api.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "pods", api.NamespaceAll, ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}, "resourceVersion": []string{"0"}})),
			rv:            "0",
			resource:      "pods",
			namespace:     api.NamespaceAll,
			fieldSelector: fields.Set{"spec.host": ""}.AsSelector(),
		},
		// pod with namespace foo and assigned field selector
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "pods", "foo", ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}, "resourceVersion": []string{"0"}})),
			rv:            "0",
			resource:      "pods",
			namespace:     "foo",
			fieldSelector: fields.Set{"spec.host": ""}.AsSelector(),
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
		client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Default.Version()})
		lw := NewListWatchFromClient(client, item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.Watch(item.rv)
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}
