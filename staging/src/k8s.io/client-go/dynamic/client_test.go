/*
Copyright 2016 The Kubernetes Authors.

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

package dynamic

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
	restclientwatch "k8s.io/client-go/rest/watch"
)

func getJSON(version, kind, name string) []byte {
	return []byte(fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "metadata": {"name": %q}}`, version, kind, name))
}

func getListJSON(version, kind string, items ...[]byte) []byte {
	json := fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "items": [%s]}`,
		version, kind, bytes.Join(items, []byte(",")))
	return []byte(json)
}

func getObject(version, kind, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": version,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"name": name,
			},
		},
	}
}

func getClientServer(h func(http.ResponseWriter, *http.Request)) (Interface, *httptest.Server, error) {
	srv := httptest.NewServer(http.HandlerFunc(h))
	cl, err := NewForConfig(&restclient.Config{
		Host: srv.URL,
	})
	if err != nil {
		srv.Close()
		return nil, nil, err
	}
	return cl, srv, nil
}

func TestList(t *testing.T) {
	tcs := []struct {
		name      string
		namespace string
		path      string
		resp      []byte
		want      *unstructured.UnstructuredList
	}{
		{
			name: "normal_list",
			path: "/apis/gtest/vtest/rtest",
			resp: getListJSON("vTest", "rTestList",
				getJSON("vTest", "rTest", "item1"),
				getJSON("vTest", "rTest", "item2")),
			want: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "vTest",
					"kind":       "rTestList",
				},
				Items: []unstructured.Unstructured{
					*getObject("vTest", "rTest", "item1"),
					*getObject("vTest", "rTest", "item2"),
				},
			},
		},
		{
			name:      "namespaced_list",
			namespace: "nstest",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest",
			resp: getListJSON("vTest", "rTestList",
				getJSON("vTest", "rTest", "item1"),
				getJSON("vTest", "rTest", "item2")),
			want: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "vTest",
					"kind":       "rTestList",
				},
				Items: []unstructured.Unstructured{
					*getObject("vTest", "rTest", "item1"),
					*getObject("vTest", "rTest", "item2"),
				},
			},
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: "rtest"}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "GET" {
				t.Errorf("List(%q) got HTTP method %s. wanted GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("List(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			w.Write(tc.resp)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Resource(resource).Namespace(tc.namespace).List(metav1.ListOptions{})
		if err != nil {
			t.Errorf("unexpected error when listing %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("List(%q) want: %v\ngot: %v", tc.name, tc.want, got)
		}
	}
}

func TestGet(t *testing.T) {
	tcs := []struct {
		resource    string
		subresource []string
		namespace   string
		name        string
		path        string
		resp        []byte
		want        *unstructured.Unstructured
	}{
		{
			resource: "rtest",
			name:     "normal_get",
			path:     "/apis/gtest/vtest/rtest/normal_get",
			resp:     getJSON("vTest", "rTest", "normal_get"),
			want:     getObject("vTest", "rTest", "normal_get"),
		},
		{
			resource:  "rtest",
			namespace: "nstest",
			name:      "namespaced_get",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_get",
			resp:      getJSON("vTest", "rTest", "namespaced_get"),
			want:      getObject("vTest", "rTest", "namespaced_get"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "normal_subresource_get",
			path:        "/apis/gtest/vtest/rtest/normal_subresource_get/srtest",
			resp:        getJSON("vTest", "srTest", "normal_subresource_get"),
			want:        getObject("vTest", "srTest", "normal_subresource_get"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			namespace:   "nstest",
			name:        "namespaced_subresource_get",
			path:        "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_subresource_get/srtest",
			resp:        getJSON("vTest", "srTest", "namespaced_subresource_get"),
			want:        getObject("vTest", "srTest", "namespaced_subresource_get"),
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: tc.resource}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "GET" {
				t.Errorf("Get(%q) got HTTP method %s. wanted GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Get(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			w.Write(tc.resp)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Resource(resource).Namespace(tc.namespace).Get(tc.name, metav1.GetOptions{}, tc.subresource...)
		if err != nil {
			t.Errorf("unexpected error when getting %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Get(%q) want: %v\ngot: %v", tc.name, tc.want, got)
		}
	}
}

func TestDelete(t *testing.T) {
	background := metav1.DeletePropagationBackground
	uid := types.UID("uid")

	statusOK := &metav1.Status{
		TypeMeta: metav1.TypeMeta{Kind: "Status"},
		Status:   metav1.StatusSuccess,
	}
	tcs := []struct {
		subresource   []string
		namespace     string
		name          string
		path          string
		deleteOptions *metav1.DeleteOptions
	}{
		{
			name: "normal_delete",
			path: "/apis/gtest/vtest/rtest/normal_delete",
		},
		{
			namespace: "nstest",
			name:      "namespaced_delete",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_delete",
		},
		{
			subresource: []string{"srtest"},
			name:        "normal_delete",
			path:        "/apis/gtest/vtest/rtest/normal_delete/srtest",
		},
		{
			subresource: []string{"srtest"},
			namespace:   "nstest",
			name:        "namespaced_delete",
			path:        "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_delete/srtest",
		},
		{
			namespace:     "nstest",
			name:          "namespaced_delete_with_options",
			path:          "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_delete_with_options",
			deleteOptions: &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}, PropagationPolicy: &background},
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: "rtest"}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "DELETE" {
				t.Errorf("Delete(%q) got HTTP method %s. wanted DELETE", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Delete(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			unstructured.UnstructuredJSONScheme.Encode(statusOK, w)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		err = cl.Resource(resource).Namespace(tc.namespace).Delete(tc.name, tc.deleteOptions, tc.subresource...)
		if err != nil {
			t.Errorf("unexpected error when deleting %q: %v", tc.name, err)
			continue
		}
	}
}

func TestDeleteCollection(t *testing.T) {
	statusOK := &metav1.Status{
		TypeMeta: metav1.TypeMeta{Kind: "Status"},
		Status:   metav1.StatusSuccess,
	}
	tcs := []struct {
		namespace string
		name      string
		path      string
	}{
		{
			name: "normal_delete_collection",
			path: "/apis/gtest/vtest/rtest",
		},
		{
			namespace: "nstest",
			name:      "namespaced_delete_collection",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest",
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: "rtest"}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "DELETE" {
				t.Errorf("DeleteCollection(%q) got HTTP method %s. wanted DELETE", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("DeleteCollection(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			unstructured.UnstructuredJSONScheme.Encode(statusOK, w)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		err = cl.Resource(resource).Namespace(tc.namespace).DeleteCollection(nil, metav1.ListOptions{})
		if err != nil {
			t.Errorf("unexpected error when deleting collection %q: %v", tc.name, err)
			continue
		}
	}
}

func TestCreate(t *testing.T) {
	tcs := []struct {
		resource    string
		subresource []string
		name        string
		namespace   string
		obj         *unstructured.Unstructured
		path        string
	}{
		{
			resource: "rtest",
			name:     "normal_create",
			path:     "/apis/gtest/vtest/rtest",
			obj:      getObject("gtest/vTest", "rTest", "normal_create"),
		},
		{
			resource:  "rtest",
			name:      "namespaced_create",
			namespace: "nstest",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest",
			obj:       getObject("gtest/vTest", "rTest", "namespaced_create"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "normal_subresource_create",
			path:        "/apis/gtest/vtest/rtest/normal_subresource_create/srtest",
			obj:         getObject("vTest", "srTest", "normal_subresource_create"),
		},
		{
			resource:    "rtest/",
			subresource: []string{"srtest"},
			name:        "namespaced_subresource_create",
			namespace:   "nstest",
			path:        "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_subresource_create/srtest",
			obj:         getObject("vTest", "srTest", "namespaced_subresource_create"),
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: tc.resource}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("Create(%q) got HTTP method %s. wanted POST", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Create(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			data, err := ioutil.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Create(%q) unexpected error reading body: %v", tc.name, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			w.Write(data)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Resource(resource).Namespace(tc.namespace).Create(tc.obj, metav1.CreateOptions{}, tc.subresource...)
		if err != nil {
			t.Errorf("unexpected error when creating %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.obj) {
			t.Errorf("Create(%q) want: %v\ngot: %v", tc.name, tc.obj, got)
		}
	}
}

func TestUpdate(t *testing.T) {
	tcs := []struct {
		resource    string
		subresource []string
		name        string
		namespace   string
		obj         *unstructured.Unstructured
		path        string
	}{
		{
			resource: "rtest",
			name:     "normal_update",
			path:     "/apis/gtest/vtest/rtest/normal_update",
			obj:      getObject("gtest/vTest", "rTest", "normal_update"),
		},
		{
			resource:  "rtest",
			name:      "namespaced_update",
			namespace: "nstest",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_update",
			obj:       getObject("gtest/vTest", "rTest", "namespaced_update"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "normal_subresource_update",
			path:        "/apis/gtest/vtest/rtest/normal_update/srtest",
			obj:         getObject("gtest/vTest", "srTest", "normal_update"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "namespaced_subresource_update",
			namespace:   "nstest",
			path:        "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_update/srtest",
			obj:         getObject("gtest/vTest", "srTest", "namespaced_update"),
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: tc.resource}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "PUT" {
				t.Errorf("Update(%q) got HTTP method %s. wanted PUT", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Update(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			data, err := ioutil.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Update(%q) unexpected error reading body: %v", tc.name, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			w.Write(data)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Resource(resource).Namespace(tc.namespace).Update(tc.obj, metav1.UpdateOptions{}, tc.subresource...)
		if err != nil {
			t.Errorf("unexpected error when updating %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.obj) {
			t.Errorf("Update(%q) want: %v\ngot: %v", tc.name, tc.obj, got)
		}
	}
}

func TestWatch(t *testing.T) {
	tcs := []struct {
		name      string
		namespace string
		events    []watch.Event
		path      string
		query     string
	}{
		{
			name:  "normal_watch",
			path:  "/apis/gtest/vtest/rtest",
			query: "watch=true",
			events: []watch.Event{
				{Type: watch.Added, Object: getObject("gtest/vTest", "rTest", "normal_watch")},
				{Type: watch.Modified, Object: getObject("gtest/vTest", "rTest", "normal_watch")},
				{Type: watch.Deleted, Object: getObject("gtest/vTest", "rTest", "normal_watch")},
			},
		},
		{
			name:      "namespaced_watch",
			namespace: "nstest",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest",
			query:     "watch=true",
			events: []watch.Event{
				{Type: watch.Added, Object: getObject("gtest/vTest", "rTest", "namespaced_watch")},
				{Type: watch.Modified, Object: getObject("gtest/vTest", "rTest", "namespaced_watch")},
				{Type: watch.Deleted, Object: getObject("gtest/vTest", "rTest", "namespaced_watch")},
			},
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: "rtest"}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "GET" {
				t.Errorf("Watch(%q) got HTTP method %s. wanted GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Watch(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}
			if r.URL.RawQuery != tc.query {
				t.Errorf("Watch(%q) got query %s. wanted %s", tc.name, r.URL.RawQuery, tc.query)
			}

			w.Header().Set("Content-Type", "application/json")

			enc := restclientwatch.NewEncoder(streaming.NewEncoder(w, unstructured.UnstructuredJSONScheme), unstructured.UnstructuredJSONScheme)
			for _, e := range tc.events {
				enc.Encode(&e)
			}
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		watcher, err := cl.Resource(resource).Namespace(tc.namespace).Watch(metav1.ListOptions{})
		if err != nil {
			t.Errorf("unexpected error when watching %q: %v", tc.name, err)
			continue
		}

		for _, want := range tc.events {
			got := <-watcher.ResultChan()
			if !reflect.DeepEqual(got, want) {
				t.Errorf("Watch(%q) want: %v\ngot: %v", tc.name, want, got)
			}
		}
	}
}

func TestPatch(t *testing.T) {
	tcs := []struct {
		resource    string
		subresource []string
		name        string
		namespace   string
		patch       []byte
		want        *unstructured.Unstructured
		path        string
	}{
		{
			resource: "rtest",
			name:     "normal_patch",
			path:     "/apis/gtest/vtest/rtest/normal_patch",
			patch:    getJSON("gtest/vTest", "rTest", "normal_patch"),
			want:     getObject("gtest/vTest", "rTest", "normal_patch"),
		},
		{
			resource:  "rtest",
			name:      "namespaced_patch",
			namespace: "nstest",
			path:      "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_patch",
			patch:     getJSON("gtest/vTest", "rTest", "namespaced_patch"),
			want:      getObject("gtest/vTest", "rTest", "namespaced_patch"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "normal_subresource_patch",
			path:        "/apis/gtest/vtest/rtest/normal_subresource_patch/srtest",
			patch:       getJSON("gtest/vTest", "srTest", "normal_subresource_patch"),
			want:        getObject("gtest/vTest", "srTest", "normal_subresource_patch"),
		},
		{
			resource:    "rtest",
			subresource: []string{"srtest"},
			name:        "namespaced_subresource_patch",
			namespace:   "nstest",
			path:        "/apis/gtest/vtest/namespaces/nstest/rtest/namespaced_subresource_patch/srtest",
			patch:       getJSON("gtest/vTest", "srTest", "namespaced_subresource_patch"),
			want:        getObject("gtest/vTest", "srTest", "namespaced_subresource_patch"),
		},
	}
	for _, tc := range tcs {
		resource := schema.GroupVersionResource{Group: "gtest", Version: "vtest", Resource: tc.resource}
		cl, srv, err := getClientServer(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "PATCH" {
				t.Errorf("Patch(%q) got HTTP method %s. wanted PATCH", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Patch(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			content := r.Header.Get("Content-Type")
			if content != string(types.StrategicMergePatchType) {
				t.Errorf("Patch(%q) got Content-Type %s. wanted %s", tc.name, content, types.StrategicMergePatchType)
			}

			data, err := ioutil.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Patch(%q) unexpected error reading body: %v", tc.name, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.Write(data)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Resource(resource).Namespace(tc.namespace).Patch(tc.name, types.StrategicMergePatchType, tc.patch, metav1.PatchOptions{}, tc.subresource...)
		if err != nil {
			t.Errorf("unexpected error when patching %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Patch(%q) want: %v\ngot: %v", tc.name, tc.want, got)
		}
	}
}
