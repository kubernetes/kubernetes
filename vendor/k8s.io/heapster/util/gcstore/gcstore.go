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

// Generic storage for garbage collected entries.
package gcstore

import (
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util"
)

var _ GCStore = &gcStore{}

// Garbage collected key-value store. Items that are not accessed after a
// certain amount of time are garbage collected. Keys must be comparable.
//
// Implementation is thread-safe.
type GCStore interface {
	// Put a key-value into the store. Keys must be comparable.
	Put(key, value interface{})

	// Gets the value of a specified key. If the key is not present,
	// nil is returned.
	Get(key interface{}) interface{}
}

func New(ttl time.Duration) GCStore {
	store := &gcStore{
		data: make(map[interface{}]*dataItem),
		ttl:  ttl,
	}
	go util.Forever(store.garbageCollect, ttl/2)
	return store
}

type gcStore struct {
	// Holds mapping of keys to values.
	data map[interface{}]*dataItem

	// TODO(vmarmol): Consider using other locking mechanism if
	// there is too much contention.
	// Synchronizes access to data.
	lock sync.Mutex

	// Time since last access for which to keep an item.
	ttl time.Duration
}

type dataItem struct {
	lastAccessed time.Time
	value        interface{}
}

func (self *gcStore) Put(key, value interface{}) {
	self.lock.Lock()
	defer self.lock.Unlock()

	self.data[key] = &dataItem{
		lastAccessed: time.Now(),
		value:        value,
	}
}

func (self *gcStore) Get(key interface{}) interface{} {
	self.lock.Lock()
	defer self.lock.Unlock()

	item, ok := self.data[key]
	if !ok {
		return nil
	}

	item.lastAccessed = time.Now()
	return item.value
}

func (self *gcStore) garbageCollect() {
	self.lock.Lock()
	defer self.lock.Unlock()

	// Remove all entried older than the allowed time.
	oldest := time.Now().Add(-self.ttl)
	for key := range self.data {
		if self.data[key].lastAccessed.Before(oldest) {
			delete(self.data, key)
		}
	}

}
