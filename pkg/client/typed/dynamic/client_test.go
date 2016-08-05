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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/streaming"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

func getJSON(version, kind, name string) []byte {
	return []byte(fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "metadata": {"name": %q}}`, version, kind, name))
}

func getListJSON(version, kind string, items ...[]byte) []byte {
	json := fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "items": [%s]}`,
		version, kind, bytes.Join(items, []byte(",")))
	return []byte(json)
}

func getObject(version, kind, name string) *runtime.Unstructured {
	return &runtime.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": version,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"name": name,
			},
		},
	}
}

func getClientServer(gv *unversioned.GroupVersion, h func(http.ResponseWriter, *http.Request)) (*Client, *httptest.Server, error) {
	srv := httptest.NewServer(http.HandlerFunc(h))
	cl, err := NewClient(&restclient.Config{
		Host:          srv.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: gv},
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
		want      *runtime.UnstructuredList
	}{
		{
			name: "normal_list",
			path: "/api/gtest/vtest/rtest",
			resp: getListJSON("vTest", "rTestList",
				getJSON("vTest", "rTest", "item1"),
				getJSON("vTest", "rTest", "item2")),
			want: &runtime.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "vTest",
					"kind":       "rTestList",
				},
				Items: []*runtime.Unstructured{
					getObject("vTest", "rTest", "item1"),
					getObject("vTest", "rTest", "item2"),
				},
			},
		},
		{
			name:      "namespaced_list",
			namespace: "nstest",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest",
			resp: getListJSON("vTest", "rTestList",
				getJSON("vTest", "rTest", "item1"),
				getJSON("vTest", "rTest", "item2")),
			want: &runtime.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "vTest",
					"kind":       "rTestList",
				},
				Items: []*runtime.Unstructured{
					getObject("vTest", "rTest", "item1"),
					getObject("vTest", "rTest", "item2"),
				},
			},
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
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

		got, err := cl.Resource(resource, tc.namespace).List(&v1.ListOptions{})
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
		namespace string
		name      string
		path      string
		resp      []byte
		want      *runtime.Unstructured
	}{
		{
			name: "normal_get",
			path: "/api/gtest/vtest/rtest/normal_get",
			resp: getJSON("vTest", "rTest", "normal_get"),
			want: getObject("vTest", "rTest", "normal_get"),
		},
		{
			namespace: "nstest",
			name:      "namespaced_get",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest/namespaced_get",
			resp:      getJSON("vTest", "rTest", "namespaced_get"),
			want:      getObject("vTest", "rTest", "namespaced_get"),
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
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

		got, err := cl.Resource(resource, tc.namespace).Get(tc.name)
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
	statusOK := &unversioned.Status{
		TypeMeta: unversioned.TypeMeta{Kind: "Status"},
		Status:   unversioned.StatusSuccess,
	}
	tcs := []struct {
		namespace string
		name      string
		path      string
	}{
		{
			name: "normal_delete",
			path: "/api/gtest/vtest/rtest/normal_delete",
		},
		{
			namespace: "nstest",
			name:      "namespaced_delete",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest/namespaced_delete",
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "DELETE" {
				t.Errorf("Delete(%q) got HTTP method %s. wanted DELETE", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Delete(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			runtime.UnstructuredJSONScheme.Encode(statusOK, w)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		err = cl.Resource(resource, tc.namespace).Delete(tc.name, nil)
		if err != nil {
			t.Errorf("unexpected error when deleting %q: %v", tc.name, err)
			continue
		}
	}
}

func TestDeleteCollection(t *testing.T) {
	statusOK := &unversioned.Status{
		TypeMeta: unversioned.TypeMeta{Kind: "Status"},
		Status:   unversioned.StatusSuccess,
	}
	tcs := []struct {
		namespace string
		name      string
		path      string
	}{
		{
			name: "normal_delete_collection",
			path: "/api/gtest/vtest/rtest",
		},
		{
			namespace: "nstest",
			name:      "namespaced_delete_collection",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest",
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "DELETE" {
				t.Errorf("DeleteCollection(%q) got HTTP method %s. wanted DELETE", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("DeleteCollection(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			runtime.UnstructuredJSONScheme.Encode(statusOK, w)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		err = cl.Resource(resource, tc.namespace).DeleteCollection(nil, &v1.ListOptions{})
		if err != nil {
			t.Errorf("unexpected error when deleting collection %q: %v", tc.name, err)
			continue
		}
	}
}

func TestCreate(t *testing.T) {
	tcs := []struct {
		name      string
		namespace string
		obj       *runtime.Unstructured
		path      string
	}{
		{
			name: "normal_create",
			path: "/api/gtest/vtest/rtest",
			obj:  getObject("vTest", "rTest", "normal_create"),
		},
		{
			name:      "namespaced_create",
			namespace: "nstest",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest",
			obj:       getObject("vTest", "rTest", "namespaced_create"),
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
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

		got, err := cl.Resource(resource, tc.namespace).Create(tc.obj)
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
		name      string
		namespace string
		obj       *runtime.Unstructured
		path      string
	}{
		{
			name: "normal_update",
			path: "/api/gtest/vtest/rtest/normal_update",
			obj:  getObject("vTest", "rTest", "normal_update"),
		},
		{
			name:      "namespaced_update",
			namespace: "nstest",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest/namespaced_update",
			obj:       getObject("vTest", "rTest", "namespaced_update"),
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
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

		got, err := cl.Resource(resource, tc.namespace).Update(tc.obj)
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
	}{
		{
			name: "normal_watch",
			path: "/api/gtest/vtest/watch/rtest",
			events: []watch.Event{
				{Type: watch.Added, Object: getObject("vTest", "rTest", "normal_watch")},
				{Type: watch.Modified, Object: getObject("vTest", "rTest", "normal_watch")},
				{Type: watch.Deleted, Object: getObject("vTest", "rTest", "normal_watch")},
			},
		},
		{
			name:      "namespaced_watch",
			namespace: "nstest",
			path:      "/api/gtest/vtest/watch/namespaces/nstest/rtest",
			events: []watch.Event{
				{Type: watch.Added, Object: getObject("vTest", "rTest", "namespaced_watch")},
				{Type: watch.Modified, Object: getObject("vTest", "rTest", "namespaced_watch")},
				{Type: watch.Deleted, Object: getObject("vTest", "rTest", "namespaced_watch")},
			},
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "GET" {
				t.Errorf("Watch(%q) got HTTP method %s. wanted GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Watch(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			enc := versioned.NewEncoder(streaming.NewEncoder(w, dynamicCodec{}), dynamicCodec{})
			for _, e := range tc.events {
				enc.Encode(&e)
			}
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		watcher, err := cl.Resource(resource, tc.namespace).Watch(&v1.ListOptions{})
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
		name      string
		namespace string
		patch     []byte
		want      *runtime.Unstructured
		path      string
	}{
		{
			name:  "normal_patch",
			path:  "/api/gtest/vtest/rtest/normal_patch",
			patch: getJSON("vTest", "rTest", "normal_patch"),
			want:  getObject("vTest", "rTest", "normal_patch"),
		},
		{
			name:      "namespaced_patch",
			namespace: "nstest",
			path:      "/api/gtest/vtest/namespaces/nstest/rtest/namespaced_patch",
			patch:     getJSON("vTest", "rTest", "namespaced_patch"),
			want:      getObject("vTest", "rTest", "namespaced_patch"),
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "gtest", Version: "vtest"}
		resource := &unversioned.APIResource{Name: "rtest", Namespaced: len(tc.namespace) != 0}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "PATCH" {
				t.Errorf("Patch(%q) got HTTP method %s. wanted PATCH", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Patch(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			content := r.Header.Get("Content-Type")
			if content != string(api.StrategicMergePatchType) {
				t.Errorf("Patch(%q) got Content-Type %s. wanted %s", tc.name, content, api.StrategicMergePatchType)
			}

			data, err := ioutil.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Patch(%q) unexpected error reading body: %v", tc.name, err)
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

		got, err := cl.Resource(resource, tc.namespace).Patch(tc.name, api.StrategicMergePatchType, tc.patch)
		if err != nil {
			t.Errorf("unexpected error when patching %q: %v", tc.name, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Patch(%q) want: %v\ngot: %v", tc.name, tc.want, got)
		}
	}
}
