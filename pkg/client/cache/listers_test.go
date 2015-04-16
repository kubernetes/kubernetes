/*
Copyright 2015 Google Inc. All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestStoreToMinionLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	ids := util.NewStringSet("foo", "bar", "baz")
	for id := range ids {
		store.Add(&api.Node{ObjectMeta: api.ObjectMeta{Name: id}})
	}
	sml := StoreToNodeLister{store}

	gotNodes, err := sml.List()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	got := make([]string, len(gotNodes.Items))
	for ix := range gotNodes.Items {
		got[ix] = gotNodes.Items[ix].Name
	}
	if !ids.HasAll(got...) || len(got) != len(ids) {
		t.Errorf("Expected %v, got %v", ids, got)
	}
}

func TestStoreToPodLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	ids := []string{"foo", "bar", "baz"}
	for _, id := range ids {
		store.Add(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   id,
				Labels: map[string]string{"name": id},
			},
		})
	}
	spl := StoreToPodLister{store}

	for _, id := range ids {
		got, err := spl.List(labels.Set{"name": id}.AsSelector())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := 1, len(got); e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}
		if e, a := id, got[0].Name; e != a {
			t.Errorf("Expected %v, got %v", e, a)
			continue
		}

		exists, err := spl.Exists(&api.Pod{ObjectMeta: api.ObjectMeta{Name: id}})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !exists {
			t.Errorf("exists returned false for %v", id)
		}
	}

	exists, err := spl.Exists(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "qux"}})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if exists {
		t.Errorf("Unexpected pod exists")
	}
}

func TestStoreToServiceLister(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	store.Add(&api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.ServiceSpec{
			Selector: map[string]string{},
		},
	})
	store.Add(&api.Service{ObjectMeta: api.ObjectMeta{Name: "bar"}})
	ssl := StoreToServiceLister{store}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   "foopod",
			Labels: map[string]string{"role": "foo"},
		},
	}

	services, err := ssl.GetPodServices(pod)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(services) != 1 {
		t.Fatalf("Expected 1 service, got %v", len(services))
	}
	if e, a := "foo", services[0].Name; e != a {
		t.Errorf("Expected service %q, got %q", e, a)
	}
}
