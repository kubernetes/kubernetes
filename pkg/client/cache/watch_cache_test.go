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

package cache

import (
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
)

func makeTestPod(name string, resourceVersion uint64) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace:       "ns",
			Name:            name,
			ResourceVersion: strconv.FormatUint(resourceVersion, 10),
		},
	}
}

func TestWatchCacheBasic(t *testing.T) {
	store := NewWatchCache(2)

	// Test Add/Update/Delete.
	pod1 := makeTestPod("pod", 1)
	if err := store.Add(pod1); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod1); !ok {
		t.Errorf("didn't find pod")
	} else {
		if !api.Semantic.DeepEqual(pod1, item) {
			t.Errorf("expected %v, got %v", pod1, item)
		}
	}
	pod2 := makeTestPod("pod", 2)
	if err := store.Update(pod2); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if item, ok, _ := store.Get(pod2); !ok {
		t.Errorf("didn't find pod")
	} else {
		if !api.Semantic.DeepEqual(pod2, item) {
			t.Errorf("expected %v, got %v", pod1, item)
		}
	}
	pod3 := makeTestPod("pod", 3)
	if err := store.Delete(pod3); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if _, ok, _ := store.Get(pod3); ok {
		t.Errorf("found pod")
	}

	// Test List.
	store.Add(makeTestPod("pod1", 4))
	store.Add(makeTestPod("pod2", 5))
	store.Add(makeTestPod("pod3", 6))
	{
		podNames := util.StringSet{}
		for _, item := range store.List() {
			podNames.Insert(item.(*api.Pod).ObjectMeta.Name)
		}
		if !podNames.HasAll("pod1", "pod2", "pod3") {
			t.Errorf("missing pods, found %v", podNames)
		}
		if len(podNames) != 3 {
			t.Errorf("found missing/extra items")
		}
	}

	// Test Replace.
	store.Replace([]interface{}{
		makeTestPod("pod4", 7),
		makeTestPod("pod5", 8),
	})
	{
		podNames := util.StringSet{}
		for _, item := range store.List() {
			podNames.Insert(item.(*api.Pod).ObjectMeta.Name)
		}
		if !podNames.HasAll("pod4", "pod5") {
			t.Errorf("missing pods, found %v", podNames)
		}
		if len(podNames) != 2 {
			t.Errorf("found missing/extra items")
		}
	}
}

func TestEvents(t *testing.T) {
	store := NewWatchCache(5)

	store.Add(makeTestPod("pod", 2))
	store.Update(makeTestPod("pod", 3))
	store.Update(makeTestPod("pod", 4))

	// Test with not full cache.
	{
		_, err := store.GetAllEventsSince(1)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.GetAllEventsSince(3)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 2 {
			t.Fatalf("unexpected events: %v", result)
		}
		for i := 0; i < 2; i++ {
			pod := makeTestPod("pod", uint64(i+3))
			if !api.Semantic.DeepEqual(pod, result[i].Object) {
				t.Errorf("unexpected item: %v, expected: %v", result[i].Object, pod)
			}
		}
	}

	for i := 5; i < 9; i++ {
		store.Update(makeTestPod("pod", uint64(i)))
	}

	// Test with full cache - there should be elements from 4 to 8.
	{
		_, err := store.GetAllEventsSince(3)
		if err == nil {
			t.Errorf("expected error too old")
		}
	}
	{
		result, err := store.GetAllEventsSince(4)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(result) != 5 {
			t.Fatalf("unexpected events: %v", result)
		}
		for i := 0; i < 5; i++ {
			pod := makeTestPod("pod", uint64(i+4))
			if !api.Semantic.DeepEqual(pod, result[i].Object) {
				t.Errorf("unexpected item: %v, expected: %v", result[i].Object, pod)
			}
		}
	}
}
