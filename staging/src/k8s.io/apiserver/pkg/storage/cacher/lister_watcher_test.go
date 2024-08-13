/*
Copyright 2023 The Kubernetes Authors.

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

package cacher

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/apis/example"
)

func TestCacherListerWatcher(t *testing.T) {
	prefix := "pods"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	objects := []*example.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}},
	}
	for _, obj := range objects {
		out := &example.Pod{}
		key := computePodKey(obj)
		if err := store.Create(context.Background(), key, obj, out, 0); err != nil {
			t.Fatalf("Create failed: %v", err)
		}
	}

	lw := NewListerWatcher(store, prefix, fn, nil)

	obj, err := lw.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	pl, ok := obj.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", pl)
	}
	if len(pl.Items) != 3 {
		t.Errorf("Expected PodList of length 3 but got %d", len(pl.Items))
	}
}

func TestCacherListerWatcherPagination(t *testing.T) {
	prefix := "pods"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	// We need the list to be sorted by name to later check the alphabetical order of
	// returned results.
	objects := []*example.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test-ns"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test-ns"}},
	}
	for _, obj := range objects {
		out := &example.Pod{}
		key := computePodKey(obj)
		if err := store.Create(context.Background(), key, obj, out, 0); err != nil {
			t.Fatalf("Create failed: %v", err)
		}
	}

	lw := NewListerWatcher(store, prefix, fn, nil)

	obj1, err := lw.List(metav1.ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit1, ok := obj1.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit1)
	}
	if len(limit1.Items) != 2 {
		t.Errorf("Expected PodList of length 2 but got %d", len(limit1.Items))
	}
	if limit1.Continue == "" {
		t.Errorf("Expected list to have Continue but got none")
	}
	obj2, err := lw.List(metav1.ListOptions{Limit: 2, Continue: limit1.Continue})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit2, ok := obj2.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit2)
	}
	if limit2.Continue != "" {
		t.Errorf("Expected list not to have Continue, but got %s", limit1.Continue)
	}

	if limit1.Items[0].Name != objects[0].Name {
		t.Errorf("Expected list1.Items[0] to be %s but got %s", objects[0].Name, limit1.Items[0].Name)
	}
	if limit1.Items[1].Name != objects[1].Name {
		t.Errorf("Expected list1.Items[1] to be %s but got %s", objects[1].Name, limit1.Items[1].Name)
	}
	if limit2.Items[0].Name != objects[2].Name {
		t.Errorf("Expected list2.Items[0] to be %s but got %s", objects[2].Name, limit2.Items[0].Name)
	}

}
