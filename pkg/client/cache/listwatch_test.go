/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/fields"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
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
	fieldSelectorQueryParamName := metav1.FieldSelectorQueryParam(api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String())
	table := []struct {
		location      string
		resource      string
		namespace     string
		fieldSelector fields.Selector
	}{
		// Node
		{
			location:      testapi.Default.ResourcePath("nodes", v1.NamespaceAll, ""),
			resource:      "nodes",
			namespace:     v1.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				testapi.Default.ResourcePath("pods", v1.NamespaceAll, ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}})),
			resource:      "pods",
			namespace:     v1.NamespaceAll,
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
		handler := utiltesting.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
		lw := NewListWatchFromClient(client.Core().RESTClient(), item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.List(v1.ListOptions{})
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

func TestListWatchesCanWatch(t *testing.T) {
	fieldSelectorQueryParamName := metav1.FieldSelectorQueryParam(api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String())
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
				testapi.Default.ResourcePathWithPrefix("watch", "nodes", v1.NamespaceAll, ""),
				buildQueryValues(url.Values{})),
			rv:            "",
			resource:      "nodes",
			namespace:     v1.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "nodes", v1.NamespaceAll, ""),
				buildQueryValues(url.Values{"resourceVersion": []string{"42"}})),
			rv:            "42",
			resource:      "nodes",
			namespace:     v1.NamespaceAll,
			fieldSelector: parseSelectorOrDie(""),
		},
		// pod with "assigned" field selector.
		{
			location: buildLocation(
				testapi.Default.ResourcePathWithPrefix("watch", "pods", v1.NamespaceAll, ""),
				buildQueryValues(url.Values{fieldSelectorQueryParamName: []string{"spec.host="}, "resourceVersion": []string{"0"}})),
			rv:            "0",
			resource:      "pods",
			namespace:     v1.NamespaceAll,
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
		handler := utiltesting.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
		lw := NewListWatchFromClient(client.Core().RESTClient(), item.resource, item.namespace, item.fieldSelector)
		// This test merely tests that the correct request is made.
		lw.Watch(v1.ListOptions{ResourceVersion: item.rv})
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

type lw struct {
	list  runtime.Object
	watch watch.Interface
}

func (w lw) List(options v1.ListOptions) (runtime.Object, error) {
	return w.list, nil
}

func (w lw) Watch(options v1.ListOptions) (watch.Interface, error) {
	return w.watch, nil
}

func TestListWatchUntil(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *v1.Pod
		fw.Modify(obj)
	}()
	listwatch := lw{
		list:  &v1.PodList{Items: []v1.Pod{{}}},
		watch: fw,
	}

	conditions := []watch.ConditionFunc{
		func(event watch.Event) (bool, error) {
			t.Logf("got %#v", event)
			return event.Type == watch.Added, nil
		},
		func(event watch.Event) (bool, error) {
			t.Logf("got %#v", event)
			return event.Type == watch.Modified, nil
		},
	}

	timeout := 10 * time.Second
	lastEvent, err := ListWatchUntil(timeout, listwatch, conditions...)
	if err != nil {
		t.Fatalf("expected nil error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Modified {
		t.Fatalf("expected MODIFIED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*v1.Pod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}
}
