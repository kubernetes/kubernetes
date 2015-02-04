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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	GetByKey(key string) (item interface{}, exists bool, err error)

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

// Index is a generic object storage interface that lets you list objects by their Index
type Index interface {
	Store
	Index(obj interface{}) ([]interface{}, error)
}

// IndexFunc knows how to provide an indexed value for an object.
type IndexFunc func(obj interface{}) (string, error)

// MetaNamespaceIndexFunc is a convenient default IndexFun which knows how to index
// an object by its namespace.
func MetaNamespaceIndexFunc(obj interface{}) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("object has no meta: %v", err)
	}
	return meta.Namespace(), nil
}

type cache struct {
	lock  sync.RWMutex
	items map[string]interface{}
	// keyFunc is used to make the key for objects stored in and retrieved from items, and
	// should be deterministic.
	keyFunc KeyFunc
	// indexFunc is used to make the index value for objects stored in an retrieved from index
	indexFunc IndexFunc
	// maps the indexFunc value for an object to a set whose keys are keys in items
	index map[string]util.StringSet
}

// Add inserts an item into the cache.
func (c *cache) Add(obj interface{}) error {
	key, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[key] = obj
	c.updateIndex(obj)
	return nil
}

// updateIndex adds or modifies an object in the index
// it is intended to be called from a function that already has a lock on the cache
func (c *cache) updateIndex(obj interface{}) error {
	if c.indexFunc == nil {
		return nil
	}
	key, err := c.keyFunc(obj)
	if err != nil {
		return err
	}
	indexValue, err := c.indexFunc(obj)
	if err != nil {
		return err
	}
	set := c.index[indexValue]
	if set == nil {
		set = util.StringSet{}
		c.index[indexValue] = set
	}
	set.Insert(key)
	return nil
}

// deleteFromIndex removes an entry from the index
// it is intended to be called from a function that already has a lock on the cache
func (c *cache) deleteFromIndex(obj interface{}) error {
	if c.indexFunc == nil {
		return nil
	}
	key, err := c.keyFunc(obj)
	if err != nil {
		return err
	}
	indexValue, err := c.indexFunc(obj)
	if err != nil {
		return err
	}
	set := c.index[indexValue]
	if set == nil {
		set = util.StringSet{}
		c.index[indexValue] = set
	}
	set.Delete(key)
	return nil
}

// Update sets an item in the cache to its updated state.
func (c *cache) Update(obj interface{}) error {
	key, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[key] = obj
	c.updateIndex(obj)
	return nil
}

// Delete removes an item from the cache.
func (c *cache) Delete(obj interface{}) error {
	key, err := c.keyFunc(obj)
	if err != nil {
		return fmt.Errorf("couldn't create key for object: %v", err)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.items, key)
	c.deleteFromIndex(obj)
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

// Index returns a list of items that match on the index function
// Index is thread-safe so long as you treat all items as immutable
func (c *cache) Index(obj interface{}) ([]interface{}, error) {
	c.lock.RLock()
	defer c.lock.RUnlock()

	indexKey, err := c.indexFunc(obj)
	if err != nil {
		return nil, err
	}
	set := c.index[indexKey]
	list := make([]interface{}, 0, set.Len())
	for _, key := range set.List() {
		list = append(list, c.items[key])
	}
	return list, nil
}

// Get returns the requested item, or sets exists=false.
// Get is completely threadsafe as long as you treat all items as immutable.
func (c *cache) Get(obj interface{}) (item interface{}, exists bool, err error) {
	key, _ := c.keyFunc(obj)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't create key for object: %v", err)
	}
	return c.GetByKey(key)
}

// GetByKey returns the request item, or exists=false.
// GetByKey is completely threadsafe as long as you treat all items as immutable.
func (c *cache) GetByKey(key string) (item interface{}, exists bool, err error) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	item, exists = c.items[key]
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

	// rebuild any index
	c.index = map[string]util.StringSet{}
	for _, item := range c.items {
		c.updateIndex(item)
	}

	return nil
}

// NewStore returns a Store implemented simply with a map and a lock.
func NewStore(keyFunc KeyFunc) Store {
	return &cache{items: map[string]interface{}{}, keyFunc: keyFunc}
}

// NewIndex returns an Index implemented simply with a map and a lock.
func NewIndex(keyFunc KeyFunc, indexFunc IndexFunc) Index {
	return &cache{items: map[string]interface{}{}, keyFunc: keyFunc, indexFunc: indexFunc, index: map[string]util.StringSet{}}
}
