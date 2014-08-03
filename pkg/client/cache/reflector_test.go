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

package cache

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestReflector_watchHandler(t *testing.T) {
	s := NewStore()
	g := NewReflector("foo", nil, &api.Pod{}, s)
	fw := watch.NewFake()
	s.Add("foo", &api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
	s.Add("bar", &api.Pod{JSONBase: api.JSONBase{ID: "bar"}})
	go func() {
		fw.Modify(&api.Pod{JSONBase: api.JSONBase{ID: "bar", ResourceVersion: 55}})
		fw.Add(&api.Pod{JSONBase: api.JSONBase{ID: "baz"}})
		fw.Add(&api.Service{JSONBase: api.JSONBase{ID: "rejected"}})
		fw.Delete(&api.Pod{JSONBase: api.JSONBase{ID: "foo"}})
		fw.Stop()
	}()
	g.watchHandler(fw)

	table := []struct {
		ID     string
		RV     uint64
		exists bool
	}{
		{"foo", 0, false},
		{"rejected", 0, false},
		{"bar", 55, true},
		{"baz", 0, true},
	}
	for _, item := range table {
		obj, exists := s.Get(item.ID)
		if e, a := item.exists, exists; e != a {
			t.Errorf("%v: expected %v, got %v", item.ID, e, a)
		}
		if !exists {
			continue
		}
		if e, a := item.RV, obj.(*api.Pod).ResourceVersion; e != a {
			t.Errorf("%v: expected %v, got %v", item.ID, e, a)
		}
	}
}

func TestReflector_startWatch(t *testing.T) {
	table := []struct{ resource, path string }{
		{"pods", "/api/v1beta1/pods/watch"},
		{"services", "/api/v1beta1/services/watch"},
	}
	for _, testItem := range table {
		got := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, req *http.Request) {
				w.WriteHeader(http.StatusNotFound)
				if req.URL.Path == testItem.path {
					close(got)
					return
				}
				t.Errorf("unexpected path %v", req.URL.Path)
			}))
		s := NewStore()
		c := client.New(srv.URL, nil)
		g := NewReflector(testItem.resource, c, &api.Pod{}, s)
		_, err := g.startWatch()
		// We're just checking that it watches the right path.
		if err == nil {
			t.Errorf("unexpected non-error")
		}
		<-got
	}
}
