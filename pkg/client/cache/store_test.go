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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Test public interface
func doTestStore(t *testing.T, store Store) {
	store.Add("foo", "bar")
	if item, ok := store.Get("foo"); !ok {
		t.Errorf("didn't find inserted item")
	} else {
		if e, a := "bar", item.(string); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
	store.Update("foo", "baz")
	if item, ok := store.Get("foo"); !ok {
		t.Errorf("didn't find inserted item")
	} else {
		if e, a := "baz", item.(string); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
	store.Delete("foo")
	if _, ok := store.Get("foo"); ok {
		t.Errorf("found deleted item??")
	}

	// Test List.
	store.Add("a", "b")
	store.Add("c", "d")
	store.Add("e", "e")
	{
		found := util.StringSet{}
		for _, item := range store.List() {
			found.Insert(item.(string))
		}
		if !found.HasAll("b", "d", "e") {
			t.Errorf("missing items")
		}
		if len(found) != 3 {
			t.Errorf("extra items")
		}

		// Check that ID list is correct.
		ids := store.ContainedIDs()
		if !ids.HasAll("a", "c", "e") {
			t.Errorf("missing items")
		}
		if len(ids) != 3 {
			t.Errorf("extra items")
		}
	}

	// Test Replace.
	store.Replace(map[string]interface{}{
		"foo": "foo",
		"bar": "bar",
	})

	{
		found := util.StringSet{}
		for _, item := range store.List() {
			found.Insert(item.(string))
		}
		if !found.HasAll("foo", "bar") {
			t.Errorf("missing items")
		}
		if len(found) != 2 {
			t.Errorf("extra items")
		}

		// Check that ID list is correct.
		ids := store.ContainedIDs()
		if !ids.HasAll("foo", "bar") {
			t.Errorf("missing items")
		}
		if len(ids) != 2 {
			t.Errorf("extra items")
		}
	}
}

func TestCache(t *testing.T) {
	doTestStore(t, NewStore())
}

func TestFIFOCache(t *testing.T) {
	doTestStore(t, NewFIFO())
}
