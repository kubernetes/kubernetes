// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gcstore

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func newTestGCStore() *gcStore {
	return &gcStore{
		data: make(map[interface{}]*dataItem),
		ttl:  time.Nanosecond,
	}
}

// Set the accessed time for testing.
func (self *gcStore) setAccessed(key interface{}, t time.Time) {
	self.lock.Lock()
	defer self.lock.Unlock()

	self.data[key].lastAccessed = t
}

// Check that the specified key has the specified value.
func checkKey(t *testing.T, store *gcStore, key string, expectedValue interface{}) {
	// Check nil case.
	if expectedValue == nil {
		actual := store.Get(key)
		assert.Equal(t, expectedValue, actual, "Expected key %q to have nil value, actual: %+v", actual)
		return
	}

	value, ok := store.Get(key).(string)
	if !ok {
		t.Errorf("Expected key %q to be a string, was: %+v", key, value)
		return
	}
	assert.Equal(t, expectedValue, value, "Expected key %q to have value %+v, actual value: %+v", key, expectedValue, value)
}

func TestBasic(t *testing.T) {
	store := newTestGCStore()

	store.Put("a", "A")
	store.Put("b", "B")

	checkKey(t, store, "a", "A")
	checkKey(t, store, "b", "B")
	checkKey(t, store, "c", nil)
}

func TestGarbageCollection(t *testing.T) {
	store := newTestGCStore()

	store.Put("a", "A")

	// Long TTL, does not GC.
	store.ttl = time.Hour
	store.garbageCollect()
	checkKey(t, store, "a", "A")

	// Short TTL, does GC
	store.ttl = 0
	store.garbageCollect()
	checkKey(t, store, "a", nil)

	// Simulate both being very old.
	now := time.Now()
	store.Put("a", "A")
	store.Put("b", "B")
	store.ttl = time.Hour
	store.setAccessed("a", now.Add(-2*time.Hour))
	store.setAccessed("b", now.Add(-2*time.Hour))

	// A get should set last accessed.
	checkKey(t, store, "a", "A")

	store.garbageCollect()
	checkKey(t, store, "a", "A")
	checkKey(t, store, "b", nil)
}
