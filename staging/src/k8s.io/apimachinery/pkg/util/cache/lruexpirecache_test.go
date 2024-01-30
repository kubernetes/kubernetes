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

package cache

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	testingclock "k8s.io/utils/clock/testing"
)

func expectEntry(t *testing.T, c *LRUExpireCache, key interface{}, value interface{}) {
	t.Helper()
	result, ok := c.Get(key)
	if !ok || result != value {
		t.Errorf("Expected cache[%v]: %v, got %v", key, value, result)
	}
}

func expectNotEntry(t *testing.T, c *LRUExpireCache, key interface{}) {
	t.Helper()
	if result, ok := c.Get(key); ok {
		t.Errorf("Expected cache[%v] to be empty, got %v", key, result)
	}
}

// Note: Check keys before checking individual entries, because Get() changes
// the eviction list.
func assertKeys(t *testing.T, gotKeys, wantKeys []interface{}) {
	t.Helper()
	if diff := cmp.Diff(gotKeys, wantKeys); diff != "" {
		t.Errorf("Wrong result for keys: diff (-got +want):\n%s", diff)
	}
}

func TestSimpleGet(t *testing.T) {
	c := NewLRUExpireCache(10)
	c.Add("long-lived", "12345", 10*time.Hour)

	assertKeys(t, c.Keys(), []interface{}{"long-lived"})

	expectEntry(t, c, "long-lived", "12345")
}

func TestSimpleRemove(t *testing.T) {
	c := NewLRUExpireCache(10)
	c.Add("long-lived", "12345", 10*time.Hour)
	c.Remove("long-lived")

	assertKeys(t, c.Keys(), []interface{}{})

	expectNotEntry(t, c, "long-lived")
}

func TestSimpleRemoveAll(t *testing.T) {
	c := NewLRUExpireCache(10)
	c.Add("long-lived", "12345", 10*time.Hour)
	c.Add("other-long-lived", "12345", 10*time.Hour)
	c.RemoveAll(func(k any) bool {
		return k.(string) == "long-lived"
	})

	assertKeys(t, c.Keys(), []any{"other-long-lived"})

	expectNotEntry(t, c, "long-lived")
	expectEntry(t, c, "other-long-lived", "12345")
}

func TestExpiredGet(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	c := NewLRUExpireCacheWithClock(10, fakeClock)
	c.Add("short-lived", "12345", 1*time.Millisecond)
	// ensure the entry expired
	fakeClock.Step(2 * time.Millisecond)

	// Keys() should not return expired keys.
	assertKeys(t, c.Keys(), []interface{}{})

	expectNotEntry(t, c, "short-lived")
}

func TestLRUOverflow(t *testing.T) {
	c := NewLRUExpireCache(4)
	c.Add("elem1", "1", 10*time.Hour)
	c.Add("elem2", "2", 10*time.Hour)
	c.Add("elem3", "3", 10*time.Hour)
	c.Add("elem4", "4", 10*time.Hour)
	c.Add("elem5", "5", 10*time.Hour)

	assertKeys(t, c.Keys(), []interface{}{"elem2", "elem3", "elem4", "elem5"})

	expectNotEntry(t, c, "elem1")
	expectEntry(t, c, "elem2", "2")
	expectEntry(t, c, "elem3", "3")
	expectEntry(t, c, "elem4", "4")
	expectEntry(t, c, "elem5", "5")
}

func TestAddBringsToFront(t *testing.T) {
	c := NewLRUExpireCache(4)
	c.Add("elem1", "1", 10*time.Hour)
	c.Add("elem2", "2", 10*time.Hour)
	c.Add("elem3", "3", 10*time.Hour)
	c.Add("elem4", "4", 10*time.Hour)

	c.Add("elem1", "1-new", 10*time.Hour)

	c.Add("elem5", "5", 10*time.Hour)

	assertKeys(t, c.Keys(), []interface{}{"elem3", "elem4", "elem1", "elem5"})

	expectNotEntry(t, c, "elem2")
	expectEntry(t, c, "elem1", "1-new")
	expectEntry(t, c, "elem3", "3")
	expectEntry(t, c, "elem4", "4")
	expectEntry(t, c, "elem5", "5")
}

func TestGetBringsToFront(t *testing.T) {
	c := NewLRUExpireCache(4)
	c.Add("elem1", "1", 10*time.Hour)
	c.Add("elem2", "2", 10*time.Hour)
	c.Add("elem3", "3", 10*time.Hour)
	c.Add("elem4", "4", 10*time.Hour)

	c.Get("elem1")

	c.Add("elem5", "5", 10*time.Hour)

	assertKeys(t, c.Keys(), []interface{}{"elem3", "elem4", "elem1", "elem5"})

	expectNotEntry(t, c, "elem2")
	expectEntry(t, c, "elem1", "1")
	expectEntry(t, c, "elem3", "3")
	expectEntry(t, c, "elem4", "4")
	expectEntry(t, c, "elem5", "5")
}

func TestLRUKeys(t *testing.T) {
	c := NewLRUExpireCache(4)
	c.Add("elem1", "1", 10*time.Hour)
	c.Add("elem2", "2", 10*time.Hour)
	c.Add("elem3", "3", 10*time.Hour)
	c.Add("elem4", "4", 10*time.Hour)

	assertKeys(t, c.Keys(), []interface{}{"elem1", "elem2", "elem3", "elem4"})
}
