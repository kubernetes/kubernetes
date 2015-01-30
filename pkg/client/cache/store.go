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
	"fmt"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
)

// Store is a generic object storage interface. Reflector knows how to watch a server
// and update a store. A generic store is provided, which allows Reflector to be used
// as a local caching system, and an LRU store, which allows Reflector to work like a
// queue of items yet to be processed.
//
// Store makes no assumptions about stored object identity; it is the responsibility
// of a Store implementation to provide a mechanism to correctly key objects and to
// define the contract for obtaining objects by some arbitrary key type.
type Store interface {
	Add(obj interface{}) error
	Update(obj interface{}) error
	Delete(obj interface{}) error
	List() []interface{}
	Get(obj interface{}) (item interface{}, exists bool, err error)

	// Replace will delete the contents of the store, using instead the
	// given list. Store takes ownership of the list, you should not reference
	// it after calling this function.
	Replace([]interface{}) error
}

// KeyFunc knows how to make a key from an object. Implementations should be deterministic.
type KeyFunc func(obj interface{}) (string, error)

// MetaNamespaceKeyFunc is a convenient default KeyFunc which knows how to make
// keys for API objects which implement meta.Interface.
// The key uses the format: <namespace>/<name>
func MetaNamespaceKeyFunc(obj interface{}) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("object has no meta: %v", err)
	}
	return meta.Namespace() + "/" + meta.Name(), nil
}

type cache struct {
	lock  sync.RWMutex
	items map[string]interface{}
	// keyFunc is used to make the key for objects stored in and retrieved from items, and
	// should be deterministic.
	keyFunc KeyFunc
}

// Add inserts an item into the cache.
func (c *cache) Add(obj interface{}) error {
	id, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[id] = obj
	return nil
}

// Update sets an item in the cache to its updated state.
func (c *cache) Update(obj interface{}) error {
	id, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[id] = obj
	return nil
}

// Delete removes an item from the cache.
func (c *cache) Delete(obj interface{}) error {
	id, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.items, id)
	return nil
}

// List returns a list of all the items.
// List is completely threadsafe as long as you treat all items as immutable.
func (c *cache) List() []interface{} {
	c.lock.RLock()
	defer c.lock.RUnlock()
	list := make([]interface{}, 0, len(c.items))
	for _, item := range c.items {
		list = append(list, item)
	}
	return list
}

// Get returns the requested item, or sets exists=false.
// Get is completely threadsafe as long as you treat all items as immutable.
func (c *cache) Get(obj interface{}) (item interface{}, exists bool, err error) {
	id, _ := c.keyFunc(obj)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.RLock()
	defer c.lock.RUnlock()
	item, exists = c.items[id]
	return item, exists, nil
}

// Replace will delete the contents of 'c', using instead the given list.
// 'c' takes ownership of the list, you should not reference the list again
// after calling this function.
func (c *cache) Replace(list []interface{}) error {
	items := map[string]interface{}{}
	for _, item := range list {
		key, err := c.keyFunc(item)
		if err != nil {
			return fmt.Errorf("couldn't create key for object: %v", err)
		}
		items[key] = item
	}

	c.lock.Lock()
	defer c.lock.Unlock()
	c.items = items
	return nil
}

// NewStore returns a Store implemented simply with a map and a lock.
func NewStore(keyFunc KeyFunc) Store {
	return &cache{items: map[string]interface{}{}, keyFunc: keyFunc}
}
